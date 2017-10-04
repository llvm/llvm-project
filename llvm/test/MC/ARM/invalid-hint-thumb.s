@ RUN: not llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 < %s 2>&1 | FileCheck %s

hint #240
hint #1000

@ FIXME: set the subclasses of the operand classes so that we only get one error for each.

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be an immediate in the range [0,239]
@ CHECK: note: operand must be an immediate in the range [0,15]

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be an immediate in the range [0,239]
@ CHECK: note: operand must be an immediate in the range [0,15]

