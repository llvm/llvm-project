# RUN: %python %s --sm=30 > %t.ll
# RUN: llc < %t.ll -march=nvptx64 -mcpu=sm_30 -mattr=+ptx50 | FileCheck %t.ll
# RUN: %if ptxas-sm_30 && ptxas-isa-5.0 %{ llc < %t.ll -march=nvptx64 -mcpu=sm_30 -mattr=+ptx50 | %ptxas-verify -arch=sm_30 %}

import fence

fence.main()
