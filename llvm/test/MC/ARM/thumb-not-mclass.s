@ RUN: not llvm-mc -triple=thumbv7m-apple-darwin -show-encoding < %s 2> %t
@ RUN: FileCheck < %t %s
@ RUN: not llvm-mc -triple=thumbv6m -show-encoding < %s 2> %t
@ RUN: FileCheck < %t %s
  .syntax unified
  .globl _func

@ Check that the assembler rejects thumb instructions that are not valid
@ on mclass.

@------------------------------------------------------------------------------
@ BLX (immediate)
@------------------------------------------------------------------------------
        blx _baz

@ CHECK: error: instruction requires: !armv*m

@------------------------------------------------------------------------------
@ SETEND
@------------------------------------------------------------------------------

        setend be
        setend le

@ CHECK: error: immediate operand must be in the range [0,1]
@ CHECK: error: immediate operand must be in the range [0,1]
