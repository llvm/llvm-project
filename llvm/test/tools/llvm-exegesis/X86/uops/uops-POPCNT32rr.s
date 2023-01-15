# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --benchmark-phase=assemble-measured-code -opcode-name=POPCNT32rr 2>&1 | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     - 'POPCNT32rr
