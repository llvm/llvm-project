# Check all variants of instructions supported by PTX94 on SM90a
# RUN: %python %s --ptx=94 --gpu-arch=90a > %t-ptx94-sm_90a.ll
# RUN: FileCheck %t-ptx94-sm_90a.ll < %t-ptx94-sm_90a.ll \
# RUN:           --check-prefixes=PTX94LDMATRIX-DAG
# RUN: llc < %t-ptx94-sm_90a.ll -mtriple=nvptx64 -mcpu=sm_90a -mattr=+ptx94 \
# RUN:           | FileCheck %t-ptx94-sm_90a.ll
# RUN: %if ptxas-sm_90a && ptxas-isa-9.4 %{                                  \
# RUN: llc < %t-ptx94-sm_90a.ll -mtriple=nvptx64 -mcpu=sm_90a -mattr=+ptx94 \
# RUN:           | %ptxas-verify -arch=sm_90a                              \
# RUN: %}

import wmma

wmma.main()
