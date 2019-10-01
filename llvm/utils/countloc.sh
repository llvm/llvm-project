#!/bin/sh
##===- utils/countloc.sh - Counts Lines Of Code --------------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script finds all the source code files in the source code directories
# (excluding certain things), runs "wc -l" on them to get the number of lines in
# each file and then sums up and prints the total with awk. 
#
# The script takes one optional option, -topdir, which specifies the top llvm
# source directory. If it is not specified then the llvm-config tool is 
# consulted to find top source dir.  
#
# Note that the implementation is based on llvmdo. See that script for more
# details.
##===----------------------------------------------------------------------===##

if test $# -gt 1 ; then
  if test "$1" = "-topdir" ; then
    TOPDIR="$2"
    shift; shift;
  else
    TOPDIR=`llvm-config --src-root`
  fi
fi

if test -d "$TOPDIR" ; then
  cd $TOPDIR
  ./utils/llvmdo -topdir "$TOPDIR" -dirs "include lib tools test utils examples" -code-only wc -l | awk '\
      BEGIN { loc=0; } \
      { loc += $1; } \
      END { print loc; }'
else
  echo "Can't find LLVM top directory"
fi
