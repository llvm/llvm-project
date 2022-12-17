# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --skip-measurements -opcode-name=LEA64r -repetition-mode=duplicate -max-configs-per-opcode=2 | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --skip-measurements -opcode-name=LEA64r -repetition-mode=loop -max-configs-per-opcode=2 | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     LEA64r
CHECK-NEXT: config: '0(%[[REG1:[A-Z0-9]+]], %[[REG2:[A-Z0-9]+]], 1)'

CHECK:      ---
CHECK-NEXT: mode: uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     LEA64r
CHECK-NEXT: config: '42(%[[REG3:[A-Z0-9]+]], %[[REG4:[A-Z0-9]+]], 1)'
