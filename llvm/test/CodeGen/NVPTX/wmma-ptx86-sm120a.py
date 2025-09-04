# Check all variants of instructions supported by PTX86 on SM120a
# RUN: %python %s --ptx=86 --gpu-arch=120 --aa > %t-ptx86-sm_120a.ll
# RUN: FileCheck %t-ptx86-sm_120a.ll < %t-ptx86-sm_120a.ll \
# RUN:           --check-prefixes=PTX86LDMATRIX-DAG,PTX86STMATRIX-DAG
# RUN: llc < %t-ptx86-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx86 \
# RUN:           | FileCheck %t-ptx86-sm_120a.ll
# RUN: %if ptxas-sm_120a && ptxas-isa-8.6 %{                                                  \
# RUN: llc < %t-ptx86-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx86 \
# RUN:           | %ptxas-verify -arch=sm_120a                              \
# RUN: %}

import wmma

wmma.main()
