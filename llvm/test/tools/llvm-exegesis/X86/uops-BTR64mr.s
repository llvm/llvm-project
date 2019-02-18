# RUN: llvm-exegesis -mode=uops -opcode-name=BTR64mr | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     BTR64mr
