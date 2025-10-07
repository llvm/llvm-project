REQUIRES: aarch64-registered-target
// Flakey on SVE buildbots, disabled pending invesgitation.
UNSUPPORTED: target={{.*}}

RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.obj --opcode-name=FMOVWSr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.obj > %t.s
RUN: FileCheck %s < %t.s

CHECK-NOT: ld{{[1-4]}}
CHECK-NOT: st{{[1-4]}}
