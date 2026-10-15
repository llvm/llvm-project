// Test for errors when MOVX instructions are misused.

// RUN: not llvm-mc -triple msp430 -mcpu=msp430x < %s 2>&1 | FileCheck %s

// The offset must fit in 20 bits.
  movx.a @r13, 0x100000(r12)
// CHECK: :[[@LINE-1]]:16: error: invalid operand for instruction
  movx.a @r13, -0x80001(r12)
// CHECK: :[[@LINE-1]]:16: error: invalid operand for instruction

// Symbolic offsets need 20-bit fixups, which are not supported yet.
  movx.a @r13, foo(r12)
// CHECK: :[[@LINE-1]]:16: error: invalid operand for instruction
  movx.a @r13, &foo
// CHECK: :[[@LINE-1]]:16: error: invalid operand for instruction

// Only an indirect source and an indexed or absolute destination are
// supported so far.
  movx.a r13, 0(r12)
// CHECK: :[[@LINE-1]]:10: error: invalid operand for instruction
  movx.a @r13+, 0(r12)
// CHECK: :[[@LINE-1]]:10: error: invalid operand for instruction
  movx.a @r13, r12
// CHECK: :[[@LINE-1]]:16: error: invalid operand for instruction
