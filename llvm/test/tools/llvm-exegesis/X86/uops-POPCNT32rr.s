# RUN: llvm-exegesis -mode=uops -opcode-name=POPCNT32rr 2>&1 | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     - 'POPCNT32rr
