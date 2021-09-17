# RUN: llvm-exegesis -mode=inverse_throughput -opcode-name=ADD32rr -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mode=inverse_throughput -opcode-name=ADD32rr -repetition-mode=loop | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: inverse_throughput
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32rr
CHECK: key: inverse_throughput
