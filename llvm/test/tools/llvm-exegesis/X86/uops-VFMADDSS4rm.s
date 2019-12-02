# RUN: llvm-exegesis -mode=uops -opcode-name=VFMADDSS4rm -repetition-mode=duplicate | FileCheck %s
# RUN: llvm-exegesis -mode=uops -opcode-name=VFMADDSS4rm -repetition-mode=loop | FileCheck %s

CHECK:      mode:            uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     VFMADDSS4rm
CHECK:      register_initial_values:
# FIXME: This will be changed to CHECK by the following patch that modeling MXCSR to VFMADDSS.
CHECK-NOT:      MXCSR
