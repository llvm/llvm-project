# Check all variants of instructions supported by PTX88 on SM120a
# RUN: %python %s --ptx=88 --gpu-arch=120 --aa > %t-ptx88-sm_120a.ll
# RUN: llc < %t-ptx88-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx88 \
# RUN:           | FileCheck %t-ptx88-sm_120a.ll
# RUN: %if ptxas-sm_120a && ptxas-isa-8.8 %{                                  \
# RUN: llc < %t-ptx88-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx88 \
# RUN:           | %ptxas-verify -arch=sm_120a                              \
# RUN: %}

import wmma

wmma.main()
