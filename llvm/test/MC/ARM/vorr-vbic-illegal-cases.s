@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text

        vorr.i32        d2, #0xffffffff
        vorr.i32        q2, #0xffffffff
        vorr.i32        d2, #0xabababab
        vorr.i32        q2, #0xabababab
        vorr.i16        q2, #0xabab
        vorr.i16        q2, #0xabab

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: operand must be a register in range [d0, d31]
@ CHECK: note: invalid operand for instruction
@ CHECK: vorr.i32        d2, #0xffffffff
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: vorr.i32        q2, #0xffffffff
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: operand must be a register in range [d0, d31]
@ CHECK: note: invalid operand for instruction
@ CHECK: vorr.i32        d2, #0xabababab
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: vorr.i32        q2, #0xabababab
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: vorr.i16        q2, #0xabab
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be a register in range [q0, q15]
@ CHECK: note: invalid operand for instruction
@ CHECK: vorr.i16        q2, #0xabab

        vbic.i32        d2, #0xffffffff
        vbic.i32        q2, #0xffffffff
        vbic.i32        d2, #0xabababab
        vbic.i32        q2, #0xabababab
        vbic.i16        d2, #0xabab
        vbic.i16        q2, #0xabab

@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        d2, #0xffffffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        q2, #0xffffffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        d2, #0xabababab
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        q2, #0xabababab
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i16        d2, #0xabab
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i16        q2, #0xabab

        vbic.i32        d2, #0x03ffffff
        vbic.i32        q2, #0x03ffff
        vbic.i32        d2, #0x03ff
        vbic.i32        d2, #0xff00ff
        vbic.i16        d2, #0x03ff
        vbic.i16        q2, #0xf0f0
        vbic.i16        q2, #0xf0f0f0

@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        d2, #0x03ffffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        q2, #0x03ffff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        d2, #0x03ff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i32        d2, #0xff00ff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i16        d2, #0x03ff
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i16        q2, #0xf0f0
@ CHECK: error: invalid operand for instruction
@ CHECK: vbic.i16        q2, #0xf0f0f0
