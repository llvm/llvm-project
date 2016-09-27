#!/bin/bash

[ -e $(basename $0) ] && echo "PLEASE USE THIS SCRIPT FROM ANOTHER DIR" && exit 1
SCRIPT_DIR=$(dirname $0)
EXECUTABLE_NAME_BASE=$(basename $SCRIPT_DIR)
LIBFUZZER_SRC=$(dirname $(dirname $SCRIPT_DIR))
JOBS=20

# FUZZ_CXXFLAGS=" -g -fsanitize=address -fsanitize-coverage=edge"
FUZZ_CXXFLAGS=" -g -fsanitize=address -fsanitize-coverage=trace-pc-guard,trace-cmp,trace-div,trace-gep"

get() {
  [ ! -e SRC ] && git clone https://github.com/openssl/openssl.git SRC && (cd SRC && git checkout OpenSSL_1_0_1f)
#  [ ! -e SRC ] && wget https://www.openssl.org/source/openssl-1.0.1f.tar.gz && tar xf openssl-1.0.1f.tar.gz && mv openssl-1.0.1f SRC
}
build_lib() {
  rm -rf BUILD
  cp -rf SRC BUILD
  (cd BUILD && ./config && make clean && make CC="clang $FUZZ_CXXFLAGS"  -j $JOBS)
}

get
build_lib
$LIBFUZZER_SRC/build.sh
clang++ -g $SCRIPT_DIR/target.cc -DCERT_PATH=\"$SCRIPT_DIR/\"  $FUZZ_CXXFLAGS BUILD/libssl.a BUILD/libcrypto.a libFuzzer.a -o $EXECUTABLE_NAME_BASE
