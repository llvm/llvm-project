# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --benchmark-phase=assemble-measured-code -opcode-name=VFMADDSS4rm -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --benchmark-phase=assemble-measured-code -opcode-name=VFMADDSS4rm -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     VFMADDSS4rm
CHECK:      register_initial_values:
CHECK:      MXCSR
