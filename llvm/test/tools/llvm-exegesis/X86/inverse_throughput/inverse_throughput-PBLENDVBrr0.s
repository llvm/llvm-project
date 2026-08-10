# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=inverse_throughput --benchmark-phase=assemble-measured-code -x86-disable-upper-sse-registers -opcode-name=PBLENDVBrr0 -repetition-mode=loop | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode:            inverse_throughput
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     - 'PBLENDVBrr0 [[LHS0:XMM[1-7]]] [[LHS0]] [[RHS0:XMM[0-7]]]'
CHECK-NEXT:     - 'PBLENDVBrr0 [[LHS1:XMM[1-7]]] [[LHS1]] [[RHS1:XMM[0-7]]]'
CHECK-NEXT:     - 'PBLENDVBrr0 [[LHS2:XMM[1-7]]] [[LHS2]] [[RHS2:XMM[0-7]]]'
CHECK-NEXT:     - 'PBLENDVBrr0 [[LHS3:XMM[1-7]]] [[LHS3]] [[RHS3:XMM[0-7]]]'
