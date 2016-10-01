#!/bin/bash
. $(dirname $0)/../common.sh

build_lib() {
  rm -rf BUILD
  cp -rf SRC BUILD
  (cd BUILD && make clean && CXX=clang++ CXXFLAGS="$FUZZ_CXXFLAGS"  make -j)
}

get_git_revision https://github.com/google/re2.git 499ef7eff7455ce9c9fae86111d4a77b6ac335de SRC
build_lib
build_libfuzzer
clang++ -g $SCRIPT_DIR/target.cc -I  BUILD BUILD/obj/libre2.a libFuzzer.a  $FUZZ_CXXFLAGS -o $EXECUTABLE_NAME_BASE
