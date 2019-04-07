# RUN: llvm-exegesis -mode=latency -opcode-name=CMOV32rr | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     CMOV32rr
CHECK-NEXT: config: ''
CHECK-LAST: ...
