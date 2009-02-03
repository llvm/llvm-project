#!/bin/sh
##===- tools/gccld/gccld.sh ------------------------------------*- bash -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Synopsis: This shell script is a replacement for the old "gccld" tool that
#           existed in LLVM versions before 2.0. The functionality of gccld has
#           now been moved to llvm-ld. This shell script provides backwards 
#           compatibility so build environments invoking gccld can still get 
#           link (under the covers) with llvm-ld.
#
# Syntax:   gccld OPTIONS... (see llvm-ld for details)
# 
##===----------------------------------------------------------------------===##
#
echo "gccld: This tool is deprecated, please use llvm-ld" 1>&2
TOOLDIR=@TOOLDIR@
$TOOLDIR/llvm-ld "$@"
