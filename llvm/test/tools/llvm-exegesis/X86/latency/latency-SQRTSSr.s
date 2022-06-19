# RUN: llvm-exegesis -mode=latency -opcode-name=SQRTSSr -repetition-mode=loop | FileCheck %s

# Check that the setup code for MXCSR does not crash the snippet.

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     SQRTSSr
CHECK-NEXT: config: ''
CHECK-NEXT: register_initial_values:
CHECK-NOT: crashed
CHECK-LAST: ...
