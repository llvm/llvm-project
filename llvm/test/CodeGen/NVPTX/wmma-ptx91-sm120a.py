# Check all variants of instructions supported by PTX91 on SM120a
# RUN: %python %s --ptx=91 --gpu-arch=120a > %t-ptx91-sm_120a.ll
# RUN: llc < %t-ptx91-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx91 \
# RUN:           | FileCheck %t-ptx91-sm_120a.ll
# RUN: %if ptxas-sm_120a && ptxas-isa-9.1 %{                                  \
# RUN: llc < %t-ptx91-sm_120a.ll -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx91 \
# RUN:           | %ptxas-verify -arch=sm_120a                                \
# RUN: %}

import wmma

wmma.main()
