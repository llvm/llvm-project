# RUN: llvm-exegesis -mode=uops -opcode-name=ADD32mi8 -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mode=uops -opcode-name=ADD32mi8 -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32mi8
