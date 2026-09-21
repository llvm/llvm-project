# Check all variants of instructions supported by PTX94 on SM100f
# RUN: %python %s --ptx=94 --gpu-arch=100f > %t-ptx94-sm_100f.ll
# RUN: FileCheck %t-ptx94-sm_100f.ll < %t-ptx94-sm_100f.ll \
# RUN:           --check-prefixes=PTX94LDMATRIX-DAG
# RUN: llc < %t-ptx94-sm_100f.ll -mtriple=nvptx64 -mcpu=sm_100f -mattr=+ptx94 \
# RUN:           | FileCheck %t-ptx94-sm_100f.ll
# RUN: %if ptxas-sm_100f && ptxas-isa-9.4 %{                                  \
# RUN: llc < %t-ptx94-sm_100f.ll -mtriple=nvptx64 -mcpu=sm_100f -mattr=+ptx94 \
# RUN:           | %ptxas-verify -arch=sm_100f                              \
# RUN: %}

import wmma

wmma.main()
