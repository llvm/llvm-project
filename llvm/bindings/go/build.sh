#!/bin/sh -xe

gollvmdir=$(dirname "$0")/llvm

workdir=$gollvmdir/workdir
llvmdir=$gollvmdir/../../..
llvm_builddir=$workdir/llvm_build

mkdir -p $llvm_builddir

cmake_flags="../../../../.. $@"
llvm_config="$llvm_builddir/bin/llvm-config"
llvm_go="$llvm_builddir/bin/llvm-go"

if test -n "`which ninja`" ; then
  # If Ninja is available, we can speed up the build by building only the
  # required subset of LLVM.
  (cd $llvm_builddir && cmake -G Ninja $cmake_flags)
  ninja -C $llvm_builddir llvm-config llvm-go
  llvm_components="$($llvm_go print-components)"
  llvm_buildtargets="$($llvm_config --libs $llvm_components | sed -e 's/-l//g')"
  ninja -C $llvm_builddir $llvm_buildtargets FileCheck
else
  (cd $llvm_builddir && cmake $cmake_flags)
  make -C $llvm_builddir -j4
fi

$llvm_go print-config > $gollvmdir/llvm_config.go
