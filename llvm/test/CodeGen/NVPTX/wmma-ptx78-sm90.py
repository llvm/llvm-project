# Check all variants of instructions supported by PTX78 on SM90
# RUN: %python %s --ptx=78 --gpu-arch=90 --aa > %t-ptx78-sm_90.ll
# RUN: FileCheck %t-ptx78-sm_90.ll < %t-ptx78-sm_90.ll \
# RUN:           --check-prefixes=PTX78STMATRIX-DAG
# RUN: llc < %t-ptx78-sm_90.ll -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 \
# RUN:           | FileCheck %t-ptx78-sm_90.ll
# RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{                                                  \
# RUN: llc < %t-ptx78-sm_90.ll -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 \
# RUN:           | %ptxas-verify -arch=sm_90                              \
# RUN: %}

import wmma

wmma.main()
