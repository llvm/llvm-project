REQUIRES: aarch64-registered-target
// This will sometimes fail with "Not all operands were initialized by the snippet generator for...".
UNSUPPORTED: target={{.*}}

RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.obj --opcode-name=FMOVWSr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.obj > %t.s
RUN: FileCheck %s < %t.s

// Start matching after the printed file path, as that may contain something that looks like a mnemonic.
CHECK: Disassembly of section .text:
CHECK-NOT: ld{{[1-4]}}
CHECK-NOT: st{{[1-4]}}
