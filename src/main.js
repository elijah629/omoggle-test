import './styles.css';
import { FaceLandmarker, FilesetResolver, DrawingUtils } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/vision_bundle.mjs';

const TARGETS = {
  canthalTilt: 4.25,
  eyeAspectRatio: 0.26,
  jawWidth: 0.68,
  cheekboneWidth: 1.14,
  midfaceRatio: 0.305,
  symmetry: 100,
};

const PAIRS = [[33,263],[133,362],[70,300],[63,293],[105,334],[46,276],[116,345],[123,352],[50,280],[187,411],[132,361],[174,399],[150,379],[172,397],[136,365],[171,396],[148,377],[176,401],[58,288]];
const JAW = [[172,397],[150,379],[171,396]];
const SCORE_WEIGHTS = {
  eyes: 0.12,
  jaw: 0.14,
  symmetry: 0.024,
  midface: 0.14,
  cheekbone: 0.10,
  eyeAspect: 0.08,
  harmony: 0.18,
};

const video = document.querySelector('#video');
const canvas = document.querySelector('#overlay');
const ctx = canvas.getContext('2d');
const statusEl = document.querySelector('#status');
const startButton = document.querySelector('#startButton');
const switchButton = document.querySelector('#switchButton');
const photoInput = document.querySelector('#photoInput');
const photoPreview = document.querySelector('#photoPreview');
const overallEl = document.querySelector('#overall');
const faceStatusEl = document.querySelector('#faceStatus');
const grid = document.querySelector('#metricGrid');
const rawData = document.querySelector('#rawData');

let landmarker;
let stream;
let useFrontCamera = true;
let lastVideoTime = -1;
let lastPayload = null;
let isCameraRunning = false;
let currentMode = 'VIDEO';
let photoObjectUrl = null;

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const angle = (a, b) => Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const round = (value, places = 3) => Number(value.toFixed(places));

function rotate(pt, cx, cy, degrees) {
  const rad = degrees * Math.PI / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const dx = pt.x - cx;
  const dy = pt.y - cy;
  return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
}

function bandScore(value, target, tolerance, falloff = tolerance * 2) {
  const delta = Math.abs(value - target);
  if (delta <= tolerance) return 10;
  return clamp(10 - ((delta - tolerance) / falloff) * 9, 1, 10);
}

function projectLandmarks(landmarks, videoWidth, videoHeight) {
  const aspect = videoHeight > 0 ? videoWidth / videoHeight : 1;
  const point = (idx) => ({ x: landmarks[idx].x * aspect, y: landmarks[idx].y });
  const roll = angle(point(10), point(152)) - 90;
  const nose = point(1);
  return { aspect, roll, point: (idx) => rotate(point(idx), nose.x, nose.y, -roll) };
}

function computeScore(landmarks, videoWidth, videoHeight) {
  if (!landmarks || landmarks.length <= 454) return null;
  const { point, roll } = projectLandmarks(landmarks, videoWidth, videoHeight);
  const faceHeight = distance(point(10), point(152));
  const faceWidth = distance(point(234), point(454));
  if (!faceHeight || !faceWidth) return null;

  const leftEyeWidth = distance(point(33), point(133));
  const rightEyeWidth = distance(point(362), point(263));
  const leftEyeHeight = distance(point(159), point(145));
  const rightEyeHeight = distance(point(386), point(374));
  const leftEyeAspect = leftEyeHeight / leftEyeWidth;
  const rightEyeAspect = rightEyeHeight / rightEyeWidth;
  const eyeAspectRatio = (leftEyeAspect + rightEyeAspect) / 2;
  const canthalTilt = -((angle(point(33), point(133)) + angle(point(362), point(263))) / 2);

  const maxJawPairDist = Math.max(...JAW.map(([a, b]) => distance(point(a), point(b))));
  const jawWidth = maxJawPairDist / faceHeight;
  const cheekboneWidth = faceWidth / maxJawPairDist;
  const eyeLineY = (((point(133).y + point(33).y) / 2) + ((point(362).y + point(263).y) / 2)) / 2;
  const midfaceRatio = Math.abs(point(0).y - eyeLineY) / faceHeight;
  const eyeSpacing = distance(point(133), point(362)) / faceWidth;

  let asymmetry = 0;
  for (const [a, b] of PAIRS) {
    const pa = point(a);
    const pb = point(b);
    const mirroredXDelta = Math.abs((pa.x + pb.x) / 2 - point(1).x);
    const yDelta = Math.abs(pa.y - pb.y);
    asymmetry += mirroredXDelta + yDelta;
  }
  asymmetry /= PAIRS.length;
  const symmetry = clamp(100 * (1 - asymmetry / 0.09), 0, 100);

  const subScores = {
    eyes: bandScore(canthalTilt, TARGETS.canthalTilt, 2.25, 8),
    jaw: bandScore(jawWidth, TARGETS.jawWidth, 0.10, 0.25),
    symmetry: clamp(symmetry / 10, 1, 10),
    midface: bandScore(midfaceRatio, TARGETS.midfaceRatio, 0.045, 0.12),
    cheekbone: bandScore(cheekboneWidth, TARGETS.cheekboneWidth, 0.16, 0.35),
    eyeAspect: bandScore(eyeAspectRatio, TARGETS.eyeAspectRatio, 0.055, 0.16),
    spacing: bandScore(eyeSpacing, 0.32, 0.06, 0.2),
  };
  subScores.harmony = 0.18 * subScores.jaw + 0.24 * subScores.midface + 0.18 * subScores.cheekbone + 0.16 * subScores.eyeAspect + 0.24 * subScores.spacing;
  const weighted = SCORE_WEIGHTS.eyes * subScores.eyes + SCORE_WEIGHTS.jaw * subScores.jaw + SCORE_WEIGHTS.symmetry * subScores.symmetry + SCORE_WEIGHTS.midface * subScores.midface + SCORE_WEIGHTS.cheekbone * subScores.cheekbone + SCORE_WEIGHTS.eyeAspect * subScores.eyeAspect + SCORE_WEIGHTS.harmony * subScores.harmony;
  const qualityMultiplier = estimateQuality(landmarks, roll, videoWidth, videoHeight);
  const overall = Math.round(10 * clamp(weighted * qualityMultiplier, 1.1, 10)) / 10;

  return {
    overall,
    faceStatus: qualityMultiplier > 0.9 ? 'good' : 'adjust',
    qualityMultiplier: round(qualityMultiplier),
    metrics: {
      canthalTilt: round(canthalTilt), jawWidth: round(jawWidth), symmetry: round(symmetry, 1),
      midfaceRatio: round(midfaceRatio), cheekboneWidth: round(cheekboneWidth), eyeAspectRatio: round(eyeAspectRatio),
      eyeSpacing: round(eyeSpacing), faceHeight: round(faceHeight), faceWidth: round(faceWidth), headRoll: round(roll),
    },
    subScores: Object.fromEntries(Object.entries(subScores).map(([k, v]) => [k, round(v, 1)])),
    targets: TARGETS,
  };
}

function estimateQuality(landmarks, roll) {
  const nose = landmarks[1];
  const centered = 1 - Math.min(1, Math.hypot(nose.x - 0.5, nose.y - 0.48) / 0.35);
  const rollScore = 1 - Math.min(1, Math.abs(roll) / 28);
  return clamp(0.72 + centered * 0.18 + rollScore * 0.10, 0.55, 1);
}

async function setupLandmarker(mode = 'VIDEO') {
  statusEl.textContent = 'Loading face model…';
  if (!landmarker) {
    const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm');
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: false,
      runningMode: mode,
      numFaces: 1,
    });
    currentMode = mode;
    return;
  }
  if (currentMode !== mode) {
    await landmarker.setOptions({ runningMode: mode });
    currentMode = mode;
  }
}

async function startCamera() {
  await setupLandmarker('VIDEO');
  if (stream) stream.getTracks().forEach((track) => track.stop());
  clearPhotoPreview();
  isCameraRunning = true;
  video.hidden = false;
  canvas.style.transform = 'scaleX(-1)';
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: useFrontCamera ? 'user' : 'environment', width: { ideal: 720 }, height: { ideal: 960 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  switchButton.disabled = false;
  statusEl.textContent = 'Scanning. Keep your face centered and well lit.';
  requestAnimationFrame(loop);
}

function loop() {
  if (!isCameraRunning) return;
  if (!video.videoWidth) return requestAnimationFrame(loop);
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    draw(result.faceLandmarks?.[0]);
    const payload = computeScore(result.faceLandmarks?.[0], video.videoWidth, video.videoHeight);
    if (payload) render(payload);
  }
  requestAnimationFrame(loop);
}

function draw(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;
  const drawingUtils = new DrawingUtils(ctx);
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: 'rgba(80, 255, 180, .20)', lineWidth: 1 });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#51f6b2', lineWidth: 2 });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#8ab4ff', lineWidth: 2 });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#8ab4ff', lineWidth: 2 });
}

function clearPhotoPreview() {
  photoPreview.hidden = true;
  photoPreview.removeAttribute('src');
  if (photoObjectUrl) URL.revokeObjectURL(photoObjectUrl);
  photoObjectUrl = null;
}

async function scoreUploadedPhoto(file) {
  if (!file) return;
  isCameraRunning = false;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.pause();
  video.removeAttribute('src');
  video.srcObject = null;
  video.hidden = true;
  switchButton.disabled = true;
  await setupLandmarker('IMAGE');
  if (photoObjectUrl) URL.revokeObjectURL(photoObjectUrl);
  photoObjectUrl = URL.createObjectURL(file);
  photoPreview.src = photoObjectUrl;
  photoPreview.hidden = false;
  canvas.style.transform = 'none';
  statusEl.textContent = 'Scoring uploaded photo…';
  await photoPreview.decode();
  canvas.width = photoPreview.naturalWidth;
  canvas.height = photoPreview.naturalHeight;
  const result = landmarker.detect(photoPreview);
  draw(result.faceLandmarks?.[0]);
  const payload = computeScore(result.faceLandmarks?.[0], photoPreview.naturalWidth, photoPreview.naturalHeight);
  if (payload) {
    render({ ...payload, source: { type: 'photo', name: file.name, width: photoPreview.naturalWidth, height: photoPreview.naturalHeight } });
    statusEl.textContent = 'Photo scored. Upload another photo or start the camera for live scoring.';
  } else {
    overallEl.textContent = '--';
    faceStatusEl.textContent = 'no face';
    faceStatusEl.dataset.status = 'adjust';
    grid.innerHTML = '';
    rawData.textContent = 'No face landmarks found in the uploaded photo.';
    statusEl.textContent = 'No face found. Try a clearer, front-facing photo.';
  }
}

function render(payload) {
  lastPayload = payload;
  overallEl.textContent = payload.overall.toFixed(1);
  faceStatusEl.textContent = payload.faceStatus;
  faceStatusEl.dataset.status = payload.faceStatus;
  const rows = [
    ['canthal tilt', payload.metrics.canthalTilt, '°', TARGETS.canthalTilt],
    ['jaw width', payload.metrics.jawWidth, '', TARGETS.jawWidth],
    ['symmetry', payload.metrics.symmetry, '%', TARGETS.symmetry],
    ['midface ratio', payload.metrics.midfaceRatio, '', TARGETS.midfaceRatio],
    ['cheekbone width', payload.metrics.cheekboneWidth, '', TARGETS.cheekboneWidth],
    ['eye aspect ratio', payload.metrics.eyeAspectRatio, '', TARGETS.eyeAspectRatio],
    ['eye spacing', payload.metrics.eyeSpacing, '', 0.32],
    ['quality multiplier', payload.qualityMultiplier, '×', 1],
  ];
  grid.innerHTML = rows.map(([name, value, unit, target]) => `<article class="metric"><span>${name}</span><strong>${value}${unit}</strong><small>target ${target}${unit === '°' ? '°' : ''}</small></article>`).join('');
  rawData.textContent = JSON.stringify(payload, null, 2);
}

startButton.addEventListener('click', () => startCamera().catch((error) => {
  console.error(error);
  statusEl.textContent = `Camera failed: ${error.message}`;
}));

switchButton.addEventListener('click', () => {
  useFrontCamera = !useFrontCamera;
  startCamera().catch((error) => statusEl.textContent = `Camera failed: ${error.message}`);
});

photoInput.addEventListener('change', () => {
  scoreUploadedPhoto(photoInput.files?.[0]).catch((error) => {
    console.error(error);
    statusEl.textContent = `Photo failed: ${error.message}`;
  });
});

window.__omoggleScore = () => lastPayload;
