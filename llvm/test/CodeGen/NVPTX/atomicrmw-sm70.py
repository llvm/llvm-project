# RUN: %python %s --sm=70 > %t.ll
# RUN: llc < %t.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | FileCheck %t.ll
# RUN: %if ptxas-sm_70 && ptxas-isa-6.3 %{ llc < %t.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | %ptxas-verify -arch=sm_70 %}

import atomicrmw

atomicrmw.main()
