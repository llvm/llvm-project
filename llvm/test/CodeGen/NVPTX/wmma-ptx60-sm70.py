# Check all variants of instructions supported by PTX60 on SM70
# RUN: %python %s --ptx=60 --gpu-arch=70 > %t-ptx60-sm_70.ll
# RUN: FileCheck %t-ptx60-sm_70.ll < %t-ptx60-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16
# RUN: FileCheck %t-ptx60-sm_70.ll < %t-ptx60-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,NOEXTGEOM,NOINT,NOSUBINT,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx60-sm_70.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx60 \
# RUN:           | FileCheck %t-ptx60-sm_70.ll
# RUN: %if ptxas %{                                                       \
# RUN: llc < %t-ptx60-sm_70.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx60 \
# RUN:           | %ptxas-verify -arch=sm_70                              \
# RUN: %}

import wmma

wmma.main()
