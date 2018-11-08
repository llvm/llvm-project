# RUN: llvm-exegesis -mode=uops -opcode-name=BEXTR32rm | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     BEXTR32rm
