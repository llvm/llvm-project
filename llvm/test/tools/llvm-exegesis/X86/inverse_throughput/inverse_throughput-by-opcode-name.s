# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=inverse_throughput --skip-measurements -opcode-name=ADD32rr -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=inverse_throughput --skip-measurements -opcode-name=ADD32rr -repetition-mode=loop | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: inverse_throughput
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32rr
