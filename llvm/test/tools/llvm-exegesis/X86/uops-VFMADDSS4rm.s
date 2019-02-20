# RUN: llvm-exegesis -mode=uops -opcode-name=VFMADDSS4rm | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     VFMADDSS4rm
