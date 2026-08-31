# Check all variants of instructions supported by PTX90 on SM110a
# RUN: %python %s --ptx=90 --gpu-arch=110a > %t-ptx90-sm_110a.ll
# RUN: llc < %t-ptx90-sm_110a.ll -mtriple=nvptx64 -mcpu=sm_110a -mattr=+ptx90 \
# RUN:           | FileCheck %t-ptx90-sm_110a.ll
# RUN: %if ptxas-sm_110a && ptxas-isa-9.0 %{                                  \
# RUN: llc < %t-ptx90-sm_110a.ll -mtriple=nvptx64 -mcpu=sm_110a -mattr=+ptx90 \
# RUN:           | %ptxas-verify -arch=sm_110a                              \
# RUN: %}

import wmma

wmma.main()
