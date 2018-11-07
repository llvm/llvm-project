# RUN: llvm-exegesis -mode=uops -opcode-name=ADD32mi8 | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32mi8
