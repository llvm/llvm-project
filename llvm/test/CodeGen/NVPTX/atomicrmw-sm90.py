# RUN: %python %s --sm=90 > %t.ll
# RUN: llc < %t.ll -march=nvptx64 -mcpu=sm_90 -mattr=+ptx87 | FileCheck %t.ll
# RUN: %if ptxas-sm_90 && ptxas-isa-8.7 %{ llc < %t.ll -march=nvptx64 -mcpu=sm_90 -mattr=+ptx87 | %ptxas-verify -arch=sm_90 %}

import atomicrmw

atomicrmw.main()
