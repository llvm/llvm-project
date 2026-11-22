# Check all variants of instructions supported by PTX94 on SM120f
# RUN: %python %s --ptx=94 --gpu-arch=120f > %t-ptx94-sm_120f.ll
# RUN: FileCheck %t-ptx94-sm_120f.ll < %t-ptx94-sm_120f.ll \
# RUN:           --check-prefixes=PTX94LDMATRIX-DAG
# RUN: llc < %t-ptx94-sm_120f.ll -mtriple=nvptx64 -mcpu=sm_120f -mattr=+ptx94 \
# RUN:           | FileCheck %t-ptx94-sm_120f.ll
# RUN: %if ptxas-sm_120f && ptxas-isa-9.4 %{                                  \
# RUN: llc < %t-ptx94-sm_120f.ll -mtriple=nvptx64 -mcpu=sm_120f -mattr=+ptx94 \
# RUN:           | %ptxas-verify -arch=sm_120f                              \
# RUN: %}

import wmma

wmma.main()
