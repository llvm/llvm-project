#!/bin/bash -xe

export BASE=`pwd`
export LLVM_SRC=${BASE}/llvm
export POLLY_SRC=${LLVM_SRC}/tools/polly
export CLANG_SRC=${LLVM_SRC}/tools/clang
export CLOOG_SRC=${BASE}/cloog_src
export CLOOG_INSTALL=${BASE}/cloog_install
export LLVM_BUILD=${BASE}/llvm_build

if [ -e /proc/cpuinfo ]; then
    procs=`cat /proc/cpuinfo | grep processor | wc -l`
else
    procs=1
fi

if ! test -d ${LLVM_SRC}; then
    git clone http://llvm.org/git/llvm.git ${LLVM_SRC}
fi

if ! test -d ${POLLY_SRC}; then
    git clone http://llvm.org/git/polly.git ${POLLY_SRC}
fi

if ! test -d ${CLANG_SRC}; then
    git clone http://llvm.org/git/clang.git ${CLANG_SRC}
fi

${POLLY_SRC}/utils/checkout_cloog.sh ${CLOOG_SRC}
cd ${CLOOG_SRC}

if ! test -e ${CLOOG_SRC}/config.log; then
    ./configure --prefix=${CLOOG_INSTALL}
fi
make
make install
cd ${BASE}

mkdir -p ${LLVM_BUILD}
cd ${LLVM_BUILD}

if which cmake ; then
    cmake -DCMAKE_PREFIX_PATH=${CLOOG_INSTALL} ${LLVM_SRC}
    make -j$procs -l$procs
    make polly-test
else
    ${LLVM_SRC}/configure --with-cloog=${CLOOG_INSTALL} --with-isl=${CLOOG_INSTALL}
    make -j$procs -l$procs
    make polly-test -C tools/polly/test/
fi
