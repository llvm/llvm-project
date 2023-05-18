# Check all variants of instructions supported by PTX64 on SM70+
# RUN: %python %s --ptx=64 --gpu-arch=70 > %t-ptx64-sm_70.ll
# RUN: FileCheck %t-ptx64-sm_70.ll < %t-ptx64-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,MMA
# RUN: FileCheck %t-ptx64-sm_70.ll < %t-ptx64-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,NOINT,NOSUBINT,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx64-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx64 \
# RUN:           | FileCheck %t-ptx64-sm_70.ll
# RUN: %if ptxas-10.1 %{                                                  \
# RUN:   llc < %t-ptx64-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx64 \
# RUN:           | %ptxas-verify -arch=sm_70                              \
# RUN: %}

import wmma

wmma.main()
