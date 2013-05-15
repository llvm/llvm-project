#!/bin/sh

clang -S -emit-llvm -O0 $1

SFILE=`echo $1 | sed -e 's/\.c/.s/g'`
LLFILE=`echo $1 | sed -e 's/\.c/.ll/g'`

opt -correlated-propagation -mem2reg -instcombine -loop-simplify -indvars \
-instnamer ${SFILE} -S > ${LLFILE}

rm ${SFILE}
