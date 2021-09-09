# RUN: llvm-exegesis -mode=uops -opcode-name=BEXTR32rm -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mode=uops -opcode-name=BEXTR32rm -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     BEXTR32rm
