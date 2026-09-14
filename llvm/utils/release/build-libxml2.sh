#!/usr/bin/env bash
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build libxml2 for Linux release binaries without ICU or iconv dependencies.
# Additional CMake arguments can select the compiler, ABI, or zlib settings.
# Zlib is enabled to match the Linux release configuration.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 BUILD_DIR INSTALL_DIR [CMAKE_ARGS...]" >&2
  exit 1
fi

mkdir -p "$1" "$2"
build_dir=$(realpath "$1")
install_dir=$(realpath "$2")
shift 2

version=2.15.1
archive="libxml2-$version.tar.xz"
checksum=c008bac08fd5c7b4a87f7b8a71f283fa581d80d80ff8d2efd3b26224c39bc54c

cd "$build_dir"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 \
    "https://download.gnome.org/sources/libxml2/2.15/$archive" -o "$archive"
fi
echo "$checksum  $archive" | sha256sum --check
tar -xf "$archive"

cmake -G Ninja -S "libxml2-$version" -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=OFF \
  -DLIBXML2_WITH_ICU=OFF \
  -DLIBXML2_WITH_ICONV=OFF \
  -DLIBXML2_WITH_MODULES=OFF \
  -DLIBXML2_WITH_PROGRAMS=OFF \
  -DLIBXML2_WITH_TESTS=OFF \
  -DLIBXML2_WITH_ZLIB=ON \
  "$@"
cmake --build build --parallel
cmake --install build --component development
