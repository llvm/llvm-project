# RUN: %python %s --sm=60 > %t.ll
# RUN: llc < %t.ll -march=nvptx64 -mcpu=sm_60 -mattr=+ptx50 | FileCheck %t.ll
# RUN: %if ptxas-sm_60 && ptxas-isa-5.0 %{ llc < %t.ll -march=nvptx64 -mcpu=sm_60 -mattr=+ptx50 | %ptxas-verify -arch=sm_60 %}

import atomicrmw

atomicrmw.main()
