#!/bin/bash
. $(dirname $0)/../common.sh
build_lib() {
  rm -rf BUILD
  cp -rf SRC BUILD
  (cd BUILD && ./buildconf && ./configure CC="clang $FUZZ_CXXFLAGS" &&  make -j)
}
get_git_revision https://github.com/c-ares/c-ares.git 51fbb479f7948fca2ace3ff34a15ff27e796afdd SRC
build_lib
build_libfuzzer
clang++ -g $SCRIPT_DIR/target.cc -I  BUILD BUILD/.libs/libcares.a libFuzzer.a  $FUZZ_CXXFLAGS -o $EXECUTABLE_NAME_BASE
