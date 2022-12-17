# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --skip-measurements -opcode-name=ADD32rm -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --skip-measurements -opcode-name=ADD32rm -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32rm
