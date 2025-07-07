# Check all variants of instructions supported by PTX61 on SM70
# RUN: %python %s --ptx=61 --gpu-arch=70 > %t-ptx61-sm_70.ll
# RUN: FileCheck %t-ptx61-sm_70.ll < %t-ptx61-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM
# RUN: FileCheck %t-ptx61-sm_70.ll < %t-ptx61-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,NOINT,NOSUBINT,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx61-sm_70.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx61 \
# RUN:           | FileCheck %t-ptx61-sm_70.ll
# RUN: %if ptxas-9.1 %{                                                   \
# RUN: llc < %t-ptx61-sm_70.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx61 \
# RUN:           | %ptxas-verify -arch=sm_70                              \
# RUN: %}

import wmma

wmma.main()
