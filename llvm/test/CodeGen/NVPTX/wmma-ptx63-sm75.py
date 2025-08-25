# Check all variants of instructions supported by PTX63 on SM75
# RUN: %python %s --ptx=63 --gpu-arch=75 > %t-ptx63-sm_75.ll
# RUN: FileCheck %t-ptx63-sm_75.ll < %t-ptx63-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT,SUBINT
# RUN: FileCheck %t-ptx63-sm_75.ll < %t-ptx63-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx63-sm_75.ll -mtriple=nvptx64 -mcpu=sm_75 -mattr=+ptx63 \
# RUN:           | FileCheck %t-ptx63-sm_75.ll
# RUN: %if ptxas-sm_75 && ptxas-isa-6.3 %{                                                  \
# RUN: llc < %t-ptx63-sm_75.ll -mtriple=nvptx64 -mcpu=sm_75 -mattr=+ptx63 \
# RUN:           | %ptxas-verify -arch=sm_75                              \
# RUN: %}

import wmma

wmma.main()
