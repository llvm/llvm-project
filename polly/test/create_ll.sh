#!/bin/sh -e

LLFILE=`echo $1 | sed -e 's/\.c/.ll/g'`
LLFILE_TMP=${LLFILE}.tmp
SOURCE=$1

shift

clang -c -S -emit-llvm -O3 -mllvm -disable-llvm-optzns ${SOURCE} -o ${LLFILE} "$@"

opt -correlated-propagation -mem2reg -instcombine -loop-simplify -indvars \
-instnamer ${LLFILE} -S -o ${LLFILE_TMP}

# Insert a header into the new testcase containing a sample RUN line a FIXME and
# an XFAIL. Then insert the formated C code and finally the LLVM-IR without
# attributes, the module ID or the target triple.
echo '; RUN: opt %loadPolly -analyze < %s | FileCheck %s' > ${LLFILE}
echo ';' >> ${LLFILE}
echo '; FIXME: Edit the run line and add checks!' >> ${LLFILE}
echo ';' >> ${LLFILE}
echo '; XFAIL: *' >> ${LLFILE}
echo ';' >> ${LLFILE}
clang-format ${SOURCE} | sed -e 's/^[^$]/;    &/' -e 's/^$/;/' >> ${LLFILE}
echo ';' >> ${LLFILE}

cat ${LLFILE_TMP} >> ${LLFILE}
sed -i".tmp" '/attributes .* =/d' ${LLFILE}
sed -i".tmp" -e 's/) \#[0-9]*/)/' ${LLFILE}
sed -i".tmp" '/; Function Attrs:/d' ${LLFILE}
sed -i".tmp" '/; ModuleID =/d' ${LLFILE}
sed -i".tmp" '/target triple/d' ${LLFILE}

mv ${LLFILE_TMP} ${LLFILE}
