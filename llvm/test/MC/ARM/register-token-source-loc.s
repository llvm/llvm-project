// RUN: not llvm-mc -triple armv6m--none-eabi < %s 2>&1 | FileCheck %s

// Some of these CHECK lines need to uses regexes to that the amount of
// whitespace between the start of the line and the caret is significant.

  add sp, r0, #4
// CHECK: error: invalid instruction, any one of the following would fix this:
// CHECK: note: instruction requires: thumb2
// CHECK: note: operand must be a register sp
// CHECK-NEXT: {{^  add sp, r0, #4}}
// CHECK-NEXT: {{^          \^}}
// CHECK: note: too many operands for instruction
