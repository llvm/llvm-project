#!/bin/bash

if [ -d "llvm-build" ]; then
    echo "Using existing 'llvm-build' directory..."    
else
    mkdir llvm-build
fi

cd llvm-build

if [ -d "llvm" ]; then
    echo "Using existing 'llvm' directory..."
else
    svn co --revision 176809 http://llvm.org/svn/llvm-project/llvm/trunk llvm
    ( cd llvm/tools ; svn co --revision 176809 http://llvm.org/svn/llvm-project/cfe/trunk clang )
fi

if [ ! -d "build" ]; then
    mkdir build
    cd build
    ../llvm/configure --enable-targets=x86_64,arm --build=x86_64-apple-darwin10 --disable-optimized --disable-assertions --enable-libcpp
    make -j8 clang-only DEBUG_SYMBOLS=1
    rm -rf lib projects runtime unittests utils config.*
    ( cd ./Debug/bin ; rm -rf ll* clang-check clang-tblgen count diagtool fpcmp macho-dump not opt yaml2obj FileCheck FileUpdate arcmt-test c-arcmt-test c-index-test bugpoint )
    ( cd ./tools ; rm -rf ll* clang-check clang-tblgen count diagtool fpcmp lto macho-dump not opt yaml2obj FileCheck FileUpdate arcmt-test c-arcmt-test c-index-test bugpoint )
    ( cd ./tools/clang ; rm -rf lib unittests utils )
    ( cd ./tools/clang/tools ; rm -rf arcmt-test c-arcmt-test c-index-test clang-check diagtool libclang )
    ( cd ../llvm ; rm -rf cmake configure docs examples projects *.txt *.TXT autoconf bindings test unittests utils ; find . -type d -name .svn -print0 | xargs -0 rm -rf )
    ( cd ../llvm/tools ; rm -rf *.txt bugpoint bugpoint-passes ll* lto macho-dump opt gold )
fi



