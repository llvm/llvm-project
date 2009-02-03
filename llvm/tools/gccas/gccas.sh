#!/bin/sh
##===- tools/gccas.sh ------------------------------------------*- bash -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Synopsis: This shell script is a replacement for the old "gccas" tool that
#           existed in LLVM versions before 2.0. The functionality of gccas has
#           now been moved to opt and llvm-as. This shell script provides 
#           backwards compatibility so build environments invoking gccas can
#           still get the net effect of llvm-as/opt by running gccas.
#
# Syntax:   gccas OPTIONS... [asm file]
# 
##===----------------------------------------------------------------------===##
#
echo "gccas: This tool is deprecated, please use opt" 1>&2
TOOLDIR=@TOOLDIR@
OPTOPTS="-std-compile-opts -f"
ASOPTS=""
lastwasdasho=0
for option in "$@" ; do
  option=`echo "$option" | sed 's/^--/-/'`
  case "$option" in
    -disable-opt)
       OPTOPTS="$OPTOPTS $option"
       ;;
    -disable-inlining)
       OPTOPTS="$OPTOPTS $option"
       ;;
    -verify)
       OPTOPTS="$OPTOPTS -verify-each"
       ;;
    -strip-debug)
       OPTOPTS="$OPTOPTS $option"
       ;;
    -o)
       OPTOPTS="$OPTOPTS -o"
       lastwasdasho=1
       ;;
    -disable-compression)
       # ignore
       ;;
    -traditional-format)
       # ignore
       ;;
    -*)
       OPTOPTS="$OPTOPTS $option"
       ;;
    *)
       if test $lastwasdasho -eq 1 ; then
         OPTOPTS="$OPTOPTS $option"
         lastwasdasho=0
       else
         ASOPTS="$ASOPTS $option"
       fi
       ;;
  esac
done
${TOOLDIR}/llvm-as $ASOPTS -o - | ${TOOLDIR}/opt $OPTOPTS
