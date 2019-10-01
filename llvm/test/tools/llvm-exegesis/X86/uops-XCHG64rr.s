# RUN: llvm-exegesis -mode=uops -opcode-name=XCHG64rr | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     XCHG64rr
