#!/bin/bash
. $(dirname $0)/../common.sh

build_lib() {
  rm -rf BUILD
  cp -rf SRC BUILD
  (cd BUILD && ./autogen.sh && CXX="clang++ $FUZZ_CXXFLAGS" CC="clang $FUZZ_CXXFLAGS" CCLD="clang++ $FUZZ_CXXFLAGS"  ./configure && make -j $JOBS)
}

get_git_tag git://git.gnome.org/libxml2  v2.9.2 SRC
build_lib
build_libfuzzer
clang++ -std=c++11  $SCRIPT_DIR/target.cc  $FUZZ_CXXFLAGS  -I BUILD/include BUILD/.libs/libxml2.a libFuzzer.a  -lz -o $EXECUTABLE_NAME_BASE
