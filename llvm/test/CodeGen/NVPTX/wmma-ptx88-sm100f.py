# Check all variants of instructions supported by PTX88 on SM120f
# RUN: %python %s --ptx=88 --gpu-arch=100f > %t-ptx88-sm_100f.ll
# RUN: llc < %t-ptx88-sm_100f.ll -mtriple=nvptx64 -mcpu=sm_100f -mattr=+ptx88 \
# RUN:           | FileCheck %t-ptx88-sm_100f.ll
# RUN: %if ptxas-sm_100f && ptxas-isa-8.8 %{                                  \
# RUN: llc < %t-ptx88-sm_100f.ll -mtriple=nvptx64 -mcpu=sm_100f -mattr=+ptx88 \
# RUN:           | %ptxas-verify -arch=sm_100f                              \
# RUN: %}

import wmma

wmma.main()
