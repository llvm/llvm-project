#!/bin/sh
##===- utils/getsrcs.sh - Counts Lines Of Code ---------------*- Script -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# details.
# 
##===----------------------------------------------------------------------===##
#
# This script just prints out the path names for all the source files in LLVM.
# The optional -topdir option can be used to specify the top LLVM source 
# directory. Without it, the llvm-config command is consulted to find the
# top source directory.
#
# Note that the implementation is based on llvmdo. See that script for more
# details.
##===----------------------------------------------------------------------===##

if test "$1" = "-topdir" ; then
  TOPDIR="$2"
  shift; shift;
else
  TOPDIR=`llvm-config --src-root`
fi

if test -d "$TOPDIR" ; then
  cd $TOPDIR
  ./utils/llvmdo -topdir "$TOPDIR" \
    -dirs "include lib tools utils examples projects" echo
else
  echo "Can't find LLVM top directory"
fi
