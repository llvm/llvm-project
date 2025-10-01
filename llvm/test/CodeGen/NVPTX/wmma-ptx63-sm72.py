# Check all variants of instructions supported by PTX63 on SM72
# RUN: %python %s --ptx=63 --gpu-arch=72 > %t-ptx63-sm_72.ll
# RUN: FileCheck %t-ptx63-sm_72.ll < %t-ptx63-sm_72.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT
# RUN: FileCheck %t-ptx63-sm_72.ll < %t-ptx63-sm_72.ll \
# RUN:           --check-prefixes=INTRINSICS,NOSUBINT,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx63-sm_72.ll -mtriple=nvptx64 -mcpu=sm_72 -mattr=+ptx63 \
# RUN:           | FileCheck %t-ptx63-sm_72.ll
# RUN: %if ptxas-sm_72 && ptxas-isa-6.3 %{                                                  \
# RUN: llc < %t-ptx63-sm_72.ll -mtriple=nvptx64 -mcpu=sm_72 -mattr=+ptx63 \
# RUN:           | %ptxas-verify -arch=sm_72                              \
# RUN: %}

import wmma

wmma.main()
