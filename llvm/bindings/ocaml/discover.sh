#!/bin/sh

link_libs=$($LLVM_CONFIG --system-libs $LINK_MODE --libs $@)
ld_flags="$($LLVM_CONFIG --ldflags) -lstdc++"
echo "(" > c_library_flags.sexp
echo $ld_flags >> c_library_flags.sexp
echo " " >> c_library_flags.sexp
echo $link_libs >> c_library_flags.sexp
echo ")" >> c_library_flags.sexp
