@ RUN: not llvm-mc -triple=armv7   -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7 -show-encoding < %s 2>&1 | FileCheck %s

@------------------------------------------------------------------------------
@ DMB
@------------------------------------------------------------------------------
        dmb #0x10
        dmb imaginary_scope

@ CHECK: error: immediate value out of range
@ CHECK: error: invalid operand for instruction

@------------------------------------------------------------------------------
@ DSB
@------------------------------------------------------------------------------
        dsb #0x10
        dsb imaginary_scope
@ CHECK: error: immediate value out of range
@ CHECK: error: invalid operand for instruction

@------------------------------------------------------------------------------
@ ISB
@------------------------------------------------------------------------------
        isb #0x1f
        isb imaginary_domain

@ CHECK: error: immediate value out of range
@ CHECK: error: invalid operand for instruction
