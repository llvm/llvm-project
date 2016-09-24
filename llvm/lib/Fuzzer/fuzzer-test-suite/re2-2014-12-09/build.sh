#!/bin/bash

[ -e $(basename $0) ] && echo "PLEASE USE THIS SCRIPT FROM ANOTHER DIR" && exit 1
SCRIPT_DIR=$(dirname $0)
EXECUTABLE_NAME_BASE=$(basename $SCRIPT_DIR)
LIBFUZZER_SRC=$(dirname $(dirname $SCRIPT_DIR))

FUZZ_CXXFLAGS="-O2 -g -fsanitize=address -fsanitize-coverage=trace-pc-guard,trace-cmp,trace-gep,trace-div"

get() {
  [ ! -e SRC ] && git clone https://github.com/google/re2.git SRC && (cd SRC && git reset --hard 499ef7eff7455ce9c9fae86111d4a77b6ac335de)
}
build_lib() {
  rm -rf BUILD
  cp -rf SRC BUILD
  (cd BUILD && make clean && CXX=clang++ CXXFLAGS="$FUZZ_CXXFLAGS"  make -j)
}

get
build_lib
$LIBFUZZER_SRC/build.sh
clang++ -g $SCRIPT_DIR/target.cc -I  BUILD BUILD/obj/libre2.a libFuzzer.a  $FUZZ_CXXFLAGS -o $EXECUTABLE_NAME_BASE
