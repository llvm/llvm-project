#!/bin/bash
#===-- build-docs.sh - Tag the LLVM release candidates ---------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# Build documentation for LLVM releases.
#
# Required Packages:
# * Fedora:
#   * dnf install doxygen python3-sphinx texlive-epstopdf ghostscript \
#                 ninja-build gcc-c++
#   * pip install sphinx-markdown-tables
# * Ubuntu:
#   * apt-get install doxygen sphinx-common python3-recommonmark \
#             ninja-build graphviz texlive-font-utils
#   * pip install sphinx-markdown-tables
#===------------------------------------------------------------------------===#

set -ex

builddir=docs-build
srcdir=$(readlink -f $(dirname "$(readlink -f "$0")")/../..)

usage() {
  echo "Build the documentation for an LLVM release.  This only needs to be "
  echo "done for -final releases."
  echo "usage: `basename $0`"
  echo " "
  echo " -release <num> Fetch the tarball for release <num> and build the "
  echo "                documentation from that source."
  echo " -srcdir  <dir> Path to llvm source directory with CMakeLists.txt"
  echo "                (optional) default: $srcdir"
}

package_doxygen() {

  project=$1
  proj_dir=$2
  output=${project}_doxygen-$release

  mv $builddir/$proj_dir/docs/doxygen/html $output
  tar -cJf $output.tar.xz $output
}


while [ $# -gt 0 ]; do
  case $1 in
    -release )
      shift
      release=$1
      ;;
    -srcdir )
      shift
      custom_srcdir=$1
      ;;
    * )
      echo "unknown option: $1"
      usage
      exit 1
      ;;
   esac
   shift
done

if [ -n "$release" -a -n "$custom_srcdir" ]; then
  echo "error: Cannot specify both -srcdir and -release options"
  exit 1
fi

if [ -n "$custom_srcdir" ]; then
  srcdir="$custom_srcdir"
fi

# Set default source directory if one is not supplied
if [ -n "$release" ]; then
  git_ref=llvmorg-$release
  if [ -d llvm-project ]; then
    echo "error llvm-project directory already exists"
    exit 1
  fi
  mkdir -p llvm-project
  pushd llvm-project
  curl -L https://github.com/llvm/llvm-project/archive/$git_ref.tar.gz | tar --strip-components=1 -xzf -
  popd
  srcdir="./llvm-project/llvm"
fi

cmake -G Ninja $srcdir -B $builddir \
               -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;polly;flang" \
               -DCMAKE_BUILD_TYPE=Release \
               -DLLVM_ENABLE_DOXYGEN=ON \
               -DLLVM_ENABLE_SPHINX=ON \
               -DLLVM_BUILD_DOCS=ON \
               -DLLVM_DOXYGEN_SVG=ON \
               -DSPHINX_WARNINGS_AS_ERRORS=OFF

ninja -C $builddir \
               docs-clang-html \
               docs-clang-tools-html \
               docs-flang-html \
               docs-lld-html \
               docs-llvm-html \
               docs-polly-html \
               doxygen-clang \
               doxygen-clang-tools \
               doxygen-flang \
               doxygen-llvm \
               doxygen-mlir \
               doxygen-polly

cmake -G Ninja $srcdir/../runtimes -B $builddir/runtimes-doc \
               -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
               -DLLVM_ENABLE_SPHINX=ON \
               -DSPHINX_WARNINGS_AS_ERRORS=OFF

ninja -C $builddir/runtimes-doc \
               docs-libcxx-html \

package_doxygen llvm .
package_doxygen clang tools/clang
package_doxygen clang-tools-extra tools/clang/tools/extra
package_doxygen flang tools/flang

html_dir=$builddir/html-export/

for d in docs/ tools/clang/docs/ tools/lld/docs/ tools/clang/tools/extra/docs/ tools/polly/docs/ tools/flang/docs/; do
  mkdir -p $html_dir/$d
  mv $builddir/$d/html/* $html_dir/$d/
done

# Keep the documentation for the runtimes under /projects/ to avoid breaking existing links.
for d in libcxx/docs/; do
  mkdir -p $html_dir/projects/$d
  mv $builddir/runtimes-doc/$d/html/* $html_dir/projects/$d/
done
