# Check all variants of instructions supported by PTX65 on SM75+
# RUN: %python %s --ptx=65 --gpu-arch=75 > %t-ptx65-sm_75.ll
# RUN: FileCheck %t-ptx65-sm_75.ll < %t-ptx65-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT,SUBINT,MMA,PTX65MMA,PTX65LDMATRIX
# RUN: FileCheck %t-ptx65-sm_75.ll < %t-ptx65-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS
# RUN: llc < %t-ptx65-sm_75.ll -mtriple=nvptx64 -mcpu=sm_75 -mattr=+ptx65 \
# RUN:           | FileCheck %t-ptx65-sm_75.ll
# RUN: %if ptxas-10.2 %{                                                  \
# RUN: llc < %t-ptx65-sm_75.ll -mtriple=nvptx64 -mcpu=sm_75 -mattr=+ptx65 \
# RUN:           | %ptxas-verify -arch=sm_75                              \
# RUN: %}

import wmma

wmma.main()
