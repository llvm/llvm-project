@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmov.i32        d2, #0xffffffab
@ CHECK: note: operand must be a register in range [d0, d31]
@ CHECK: note: invalid operand for instruction
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmov.i32        q2, #0xffffffab
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmov.i16        q2, #0xffab
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmov.i16        q2, #0xffab
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmvn.i32        d2, #0xffffffab
@ CHECK: note: operand must be a register in range [d0, d31]
@ CHECK: note: invalid operand for instruction
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmvn.i32        q2, #0xffffffab
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmvn.i16        q2, #0xffab
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: vmvn.i16        q2, #0xffab
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction

        vmov.i32        d2, #0xffffffab
        vmov.i32        q2, #0xffffffab
        vmov.i16        q2, #0xffab
        vmov.i16        q2, #0xffab

        vmvn.i32        d2, #0xffffffab
        vmvn.i32        q2, #0xffffffab
        vmvn.i16        q2, #0xffab
        vmvn.i16        q2, #0xffab
