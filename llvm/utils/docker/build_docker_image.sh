#!/bin/bash
#===- llvm/utils/docker/build_docker_image.sh ----------------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===//
set -e

IMAGE_SOURCE=""
DOCKER_REPOSITORY=""
DOCKER_TAG=""
BUILDSCRIPT_ARGS=""

function show_usage() {
  usage=$(cat << EOF
Usage: build_docker_image.sh [options] [-- [buildscript_args]...]

Available options:
    -s|--source             image source dir (i.e. debian8, nvidia-cuda, etc)
    -d|--docker-repository  docker repository for the image
    -t|--docker-tag         docker tag for the image
Required options: --source and --docker-repository.

All options after '--' are passed to buildscript (see
scripts/build_install_llvm.sh).

For example, running:
$ build_docker_image.sh -s debian8 -d mydocker/debian8-clang -t latest \ 
  -- -p clang -i install-clang -i install-clang-headers
will produce two docker images:
    mydocker/debian8-clang-build:latest - an intermediate image used to compile
      clang.
    mydocker/clang-debian8:latest       - a small image with preinstalled clang.
Please note that this example produces a not very useful installation, since it
doesn't override CMake defaults, which produces a Debug and non-boostrapped
version of clang.

For an example of a somewhat more useful build, search for 2-stage build
instructions in llvm/docs/Docker.rst.
EOF
)
  echo "$usage"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    -s|--source)
      shift
      IMAGE_SOURCE="$1"
      shift
      ;;
    -d|--docker-repository)
      shift
      DOCKER_REPOSITORY="$1"
      shift
      ;;
    -t|--docker-tag)
      shift
      DOCKER_TAG="$1"
      shift
      ;;
    --)
      shift
      BUILDSCRIPT_ARGS="$*"
      shift $#
      ;;
    *)
      echo "Unknown argument $1"
      exit 1
      ;;
  esac
done

command -v docker >/dev/null ||
  {
    echo "Docker binary cannot be found. Please install Docker to use this script."
    exit 1
  }

if [ "$IMAGE_SOURCE" == "" ]; then
  echo "Required argument missing: --source"
  exit 1
fi

if [ "$DOCKER_REPOSITORY" == "" ]; then
  echo "Required argument missing: --docker-repository"
  exit 1
fi

cd $(dirname $0)
if [ ! -d $IMAGE_SOURCE ]; then
  echo "No sources for '$IMAGE_SOURCE' were found in $PWD"
  exit 1
fi

echo "Building from $IMAGE_SOURCE"

if [ "$DOCKER_TAG" != "" ]; then
  DOCKER_TAG=":$DOCKER_TAG"
fi

echo "Building $DOCKER_REPOSITORY-build$DOCKER_TAG"
docker build -t "$DOCKER_REPOSITORY-build$DOCKER_TAG" \
  --build-arg "buildscript_args=$BUILDSCRIPT_ARGS" \
  -f "$IMAGE_SOURCE/build/Dockerfile" .

echo "Copying clang installation to release image sources"
docker run -v "$PWD/$IMAGE_SOURCE:/workspace" "$DOCKER_REPOSITORY-build$DOCKER_TAG" \
  cp /tmp/clang.tar.gz /workspace/release
trap "rm -f $PWD/$IMAGE_SOURCE/release/clang.tar.gz" EXIT

echo "Building release image"
docker build -t "${DOCKER_REPOSITORY}${DOCKER_TAG}" \
  "$IMAGE_SOURCE/release"

echo "Done"
