# Check all variants of instructions supported by PTX86 on SM100a
# RUN: %python %s --ptx=86 --gpu-arch=100 --aa > %t-ptx86-sm_100a.ll
# RUN: FileCheck %t-ptx86-sm_100a.ll < %t-ptx86-sm_100a.ll \
# RUN:           --check-prefixes=PTX86LDMATRIX-DAG,PTX86STMATRIX-DAG
# RUN: llc < %t-ptx86-sm_100a.ll -mtriple=nvptx64 -mcpu=sm_100a -mattr=+ptx86 \
# RUN:           | FileCheck %t-ptx86-sm_100a.ll
# RUN: %if ptxas-12.7 %{                                                  \
# RUN: llc < %t-ptx86-sm_100a.ll -mtriple=nvptx64 -mcpu=sm_100a -mattr=+ptx86 \
# RUN:           | %ptxas-verify -arch=sm_100a                              \
# RUN: %}

import wmma

wmma.main()
