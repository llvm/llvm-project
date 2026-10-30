// MSP430X instructions require -mcpu=msp430x.
// RUN: not llvm-mc -triple msp430 %s 2>&1 | FileCheck %s

  rram #1, r12
// CHECK: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  rrax r12
// CHECK: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  rlax r12
// CHECK: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
