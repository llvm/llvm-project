# RUN: llvm-exegesis -mode=latency -opcode-name=CMOV32rr -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mode=latency -opcode-name=CMOV32rr -repetition-mode=loop | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     'CMOV32rr {{.*}} i_0x{{[0-9a-f]}}'
CHECK-NEXT: config: ''
CHECK-LAST: ...
