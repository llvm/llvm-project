# Check all variants of instructions supported by PTX87 on SM120a
# RUN: %python %s --ptx=87 --gpu-arch=120 --aa > %t-ptx87-sm_120a.ll
# RUN: llc < %t-ptx87-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx87 \
# RUN:           | FileCheck %t-ptx87-sm_120a.ll
# RUN: %if ptxas-sm_120a && ptxas-isa-8.7 %{                                  \
# RUN: llc < %t-ptx87-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx87 \
# RUN:           | %ptxas-verify -arch=sm_120a                              \
# RUN: %}

import wmma

wmma.main()
