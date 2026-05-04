#!/usr/bin/env bash
#===-- ejit-minimal.sh ---------------------------------------------------===#
# Convenience script: configure and build the minimal EJIT runtime.
#
# Usage:
#   ./llvm/utils/ejit-minimal.sh           # configure + build + size check
#   ./llvm/utils/ejit-minimal.sh configure # configure only
#   ./llvm/utils/ejit-minimal.sh build     # build only (after configure)
#   ./llvm/utils/ejit-minimal.sh size      # size check only
#
# The resulting combined archive is at:
#   build-ejit-minimal/lib/ExecutionEngine/EJIT/libejit_minimal.a
#===----------------------------------------------------------------------===#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLVM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${LLVM_ROOT}/.." && pwd)"
BUILD_DIR="${WORKSPACE_ROOT}/build-ejit-minimal"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[ejit-minimal]${NC} $*"; }
warn() { echo -e "${YELLOW}[ejit-minimal]${NC} $*"; }
err()  { echo -e "${RED}[ejit-minimal]${NC} $*"; }

do_configure() {
  log "Configuring minimal EJIT build..."
  log "  Source:    ${LLVM_ROOT}"
  log "  Build dir: ${BUILD_DIR}"

  cmake -S "${LLVM_ROOT}" -B "${BUILD_DIR}" \
    --preset ejit-minimal \
    "$@"

  log "Configuration complete."
  log "Run '$0 build' to build, or 'cd ${BUILD_DIR} && ninja LLVMEJIT'"
}

do_build() {
  if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
    err "Build directory not configured. Run '$0 configure' first."
    exit 1
  fi

  log "Building LLVMEJIT + dependencies..."
  ninja -C "${BUILD_DIR}" LLVMEJIT

  log "Combining into libejit_minimal.a..."
  ninja -C "${BUILD_DIR}" ejit_minimal

  log "Build complete."
}

do_size_check() {
  if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
    err "Build directory not configured. Run '$0 configure' first."
    exit 1
  fi

  log "Running size check..."
  ninja -C "${BUILD_DIR}" check-ejit-size
}

do_clean() {
  log "Cleaning build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
}

case "${1:-all}" in
  configure)
    shift
    do_configure "$@"
    ;;
  build)
    do_build
    ;;
  size)
    do_size_check
    ;;
  clean)
    do_clean
    ;;
  all)
    do_configure
    do_build
    do_size_check
    ;;
  *)
    echo "Usage: $0 {configure|build|size|clean|all}"
    echo ""
    echo "  configure  — Run CMake with ejit-minimal preset"
    echo "  build      — Build LLVMEJIT + combine into libejit_minimal.a"
    echo "  size       — Check if combined archive fits in budget"
    echo "  clean      — Remove build directory"
    echo "  all        — configure + build + size (default)"
    exit 1
    ;;
esac
