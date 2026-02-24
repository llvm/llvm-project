# Check all variants of instructions supported by PTX91 on SM120f
# RUN: %python %s --ptx=91 --gpu-arch=120f > %t-ptx91-sm_120f.ll
# RUN: llc < %t-ptx91-sm_120f.ll -mtriple=nvptx64 -mcpu=sm_120f -mattr=+ptx91 \
# RUN:           | FileCheck %t-ptx91-sm_120f.ll
# RUN: %if ptxas-sm_120f && ptxas-isa-9.1 %{                                  \
# RUN: llc < %t-ptx91-sm_120f.ll -mtriple=nvptx64 -mcpu=sm_120f -mattr=+ptx91 \
# RUN:           | %ptxas-verify -arch=sm_120f                                \
# RUN: %}

import wmma

wmma.main()
