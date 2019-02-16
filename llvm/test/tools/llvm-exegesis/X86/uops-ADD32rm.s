# RUN: llvm-exegesis -mode=uops -opcode-name=ADD32rm | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32rm
