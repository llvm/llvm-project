#!/bin/sh

link_libs=$($LLVM_CONFIG $LINK_MODE --libs $@)
ld_flags=$($LLVM_CONFIG --ldflags)
echo "(" > c_library_flags.sexp
echo $ld_flags >> c_library_flags.sexp
echo " " >> c_library_flags.sexp
echo $link_libs >> c_library_flags.sexp
echo ")" >> c_library_flags.sexp
