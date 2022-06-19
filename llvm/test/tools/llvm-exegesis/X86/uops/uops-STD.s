# RUN: llvm-exegesis -mode=uops -opcode-name=STD -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mode=uops -opcode-name=STD -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     STD
