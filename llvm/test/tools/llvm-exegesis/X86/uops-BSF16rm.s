# RUN: llvm-exegesis -mode=uops -opcode-name=BSF16rm | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     BSF16rm
