#!/bin/sh

llvm_config=$1
shift 1
link_libs=$($llvm_config --libs $@)
ld_flags=$($llvm_config --ldflags)
echo "(" > c_library_flags.sexp
echo $ld_flags >> c_library_flags.sexp
echo " " >> c_library_flags.sexp
echo $link_libs >> c_library_flags.sexp
echo ")" >> c_library_flags.sexp
