REQUIRES: aarch64-registered-target

RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=FMOVWSr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s < %t.s

CHECK-NOT: ld{{[1-4]}}
CHECK-NOT: st{{[1-4]}}
