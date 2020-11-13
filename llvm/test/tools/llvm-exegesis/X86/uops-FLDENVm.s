# RUN: llvm-exegesis -mode=uops -opcode-name=FLDENVm,FLDL2E -repetition-mode=duplicate | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     FLDENVm
