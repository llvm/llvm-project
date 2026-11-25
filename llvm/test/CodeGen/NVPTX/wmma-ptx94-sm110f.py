# Check all variants of instructions supported by PTX94 on SM110f
# RUN: %python %s --ptx=94 --gpu-arch=110f > %t-ptx94-sm_110f.ll
# RUN: FileCheck %t-ptx94-sm_110f.ll < %t-ptx94-sm_110f.ll \
# RUN:           --check-prefixes=PTX94LDMATRIX-DAG
# RUN: llc < %t-ptx94-sm_110f.ll -mtriple=nvptx64 -mcpu=sm_110f -mattr=+ptx94 \
# RUN:           | FileCheck %t-ptx94-sm_110f.ll
# RUN: %if ptxas-sm_110f && ptxas-isa-9.4 %{                                  \
# RUN: llc < %t-ptx94-sm_110f.ll -mtriple=nvptx64 -mcpu=sm_110f -mattr=+ptx94 \
# RUN:           | %ptxas-verify -arch=sm_110f                              \
# RUN: %}

import wmma

wmma.main()
