# RUN: llvm-exegesis -mode=uops -opcode-name=ADD_F32m -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mode=uops -opcode-name=ADD_F32m -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD_F32m
CHECK:      register_initial_values:
CHECK:      FPCW
