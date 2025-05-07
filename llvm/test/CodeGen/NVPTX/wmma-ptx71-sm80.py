# Check all variants of instructions supported by PTX71 on SM80+
# RUN: %python %s --ptx=71 --gpu-arch=80 > %t-ptx71-sm_80.ll
# RUN: FileCheck %t-ptx71-sm_80.ll < %t-ptx71-sm_80.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT,SUBINT,MMA,ALTFLOAT,DOUBLE,PTX65MMA,PTX65LDMATRIX,PTX71MMA
# RUN: FileCheck %t-ptx71-sm_80.ll < %t-ptx71-sm_80.ll \
# RUN:           --check-prefixes=INTRINSICS
# RUN: llc < %t-ptx71-sm_80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx71 \
# RUN:           | FileCheck %t-ptx71-sm_80.ll
# RUN: %if ptxas-11.1 %{                                                  \
# RUN: llc < %t-ptx71-sm_80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx71 \
# RUN:           | %ptxas-verify -arch=sm_80                              \
# RUN: %}

import wmma

wmma.main()
