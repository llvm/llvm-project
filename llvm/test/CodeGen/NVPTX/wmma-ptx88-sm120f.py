# Check all variants of instructions supported by PTX88 on SM120f
# RUN: %python %s --ptx=88 --gpu-arch=120f > %t-ptx88-sm_120f.ll
# RUN: llc < %t-ptx88-sm_120f.ll -mtriple=nvptx64 -mcpu=sm_120f -mattr=+ptx88 \
# RUN:           | FileCheck %t-ptx88-sm_120f.ll
# RUN: %if ptxas-sm_120f && ptxas-isa-8.8 %{                                  \
# RUN: llc < %t-ptx88-sm_120f.ll -mtriple=nvptx64 -mcpu=sm_120f -mattr=+ptx88 \
# RUN:           | %ptxas-verify -arch=sm_120f                              \
# RUN: %}

import wmma

wmma.main()
