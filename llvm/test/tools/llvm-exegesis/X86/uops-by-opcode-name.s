# RUN: llvm-exegesis -mode=uops -opcode-name=ADD32rr | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32rr
