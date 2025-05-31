# Check all variants of instructions supported by PTX86 on SM101a
# RUN: %python %s --ptx=86 --gpu-arch=101 --aa > %t-ptx86-sm_101a.ll
# RUN: FileCheck %t-ptx86-sm_101a.ll < %t-ptx86-sm_101a.ll \
# RUN:           --check-prefixes=PTX86LDMATRIX-DAG
# RUN: FileCheck %t-ptx86-sm_101a.ll < %t-ptx86-sm_101a.ll \
# RUN:           --check-prefixes=PTX86LDMATRIX-DAG
# RUN: llc < %t-ptx86-sm_101a.ll -mtriple=nvptx64 -mcpu=sm_101a -mattr=+ptx86 \
# RUN:           | FileCheck %t-ptx86-sm_101a.ll
# RUN: %if ptxas-12.7 %{                                                  \
# RUN: llc < %t-ptx86-sm_101a.ll -mtriple=nvptx64 -mcpu=sm_101a -mattr=+ptx86 \
# RUN:           | %ptxas-verify -arch=sm_101a                              \
# RUN: %}

import wmma

wmma.main()
