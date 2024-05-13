# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --benchmark-phase=assemble-measured-code -opcode-name=ADD_F32m -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --benchmark-phase=assemble-measured-code -opcode-name=ADD_F32m -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD_F32m
CHECK:      register_initial_values:
CHECK:      FPCW
