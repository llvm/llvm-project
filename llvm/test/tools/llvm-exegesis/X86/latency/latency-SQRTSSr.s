# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=SQRTSSr -repetition-mode=loop | FileCheck %s

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
