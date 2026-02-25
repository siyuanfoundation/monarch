#!/bin/bash
set -e
export LOCATION="us-east5"
export REPO_NAME="${USER}-repo"
export PROJECT_ID=$(gcloud config get project)

gcloud auth configure-docker ${LOCATION}-docker.pkg.dev --quiet

export IMAGE_PATH=${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/monarch:my-test

rm -rf dist
USE_TENSOR_ENGINE=0 uv build --no-build-isolation --wheel

# Use a CPU-only pytorch stable tag that we confirmed is Python 3.12.
PYTORCH_CPU_TAG="2.12.0.dev20260324-cuda12.8-cudnn9-runtime"

DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile.nightly \
    -t ${IMAGE_PATH} \
    --build-context monarch-wheels=dist \
    --build-arg PYTORCH_TAG=${PYTORCH_CPU_TAG} \
    .

docker push ${IMAGE_PATH}
echo "Build and push successful: ${IMAGE_PATH}"
