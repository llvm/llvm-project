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
Usage: build_docker_image.sh [options] [-- [cmake_args]...]

Available options:
  General:
    -h|--help               show this help message
  Docker-specific:
    -s|--source             image source dir (i.e. debian8, nvidia-cuda, etc)
    -d|--docker-repository  docker repository for the image
    -t|--docker-tag         docker tag for the image
  LLVM-specific:
    -b|--branch         svn branch to checkout, i.e. 'trunk',
                        'branches/release_40'
                        (default: 'trunk')
    -r|--revision       svn revision to checkout
    -p|--llvm-project   name of an svn project to checkout. Will also add the
                        project to a list LLVM_ENABLE_PROJECTS, passed to CMake.
                        For clang, please use 'clang', not 'cfe'.
                        Project 'llvm' is always included and ignored, if
                        specified.
                        Can be specified multiple times.
    -i|--install-target name of a cmake install target to build and include in
                        the resulting archive. Can be specified multiple times.

Required options: --source and --docker-repository, at least one
  --install-target.

All options after '--' are passed to CMake invocation.

For example, running:
$ build_docker_image.sh -s debian8 -d mydocker/debian8-clang -t latest \ 
  -p clang -i install-clang -i install-clang-headers
will produce two docker images:
    mydocker/debian8-clang-build:latest - an intermediate image used to compile
      clang.
    mydocker/clang-debian8:latest       - a small image with preinstalled clang.
Please note that this example produces a not very useful installation, since it
doesn't override CMake defaults, which produces a Debug and non-boostrapped
version of clang.

To get a 2-stage clang build, you could use this command:
$ ./build_docker_image.sh -s debian8 -d mydocker/clang-debian8 -t "latest" \ 
    -p clang -i stage2-install-clang -i stage2-install-clang-headers \ 
    -- \ 
    -DLLVM_TARGETS_TO_BUILD=Native -DCMAKE_BUILD_TYPE=Release \ 
    -DBOOTSTRAP_CMAKE_BUILD_TYPE=Release \ 
    -DCLANG_ENABLE_BOOTSTRAP=ON \ 
    -DCLANG_BOOTSTRAP_TARGETS="install-clang;install-clang-headers"
EOF
)
  echo "$usage"
}

SEEN_INSTALL_TARGET=0
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
    -i|--install-target|-r|--revision|-b|--branch|-p|--llvm-project)
      if [ "$1" == "-i" ] || [ "$1" == "--install-target" ]; then
        SEEN_INSTALL_TARGET=1
      fi
      BUILDSCRIPT_ARGS="$BUILDSCRIPT_ARGS $1 $2"
      shift 2
      ;;
    --)
      shift
      BUILDSCRIPT_ARGS="$BUILDSCRIPT_ARGS -- $*"
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

if [ $SEEN_INSTALL_TARGET -eq 0 ]; then
  echo "Please provide at least one --install-target"
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
