@ RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi < %s 2> %t
@ RUN: FileCheck < %t %s

@ Check the diagnostics for .cantunwind, .handlerdata, and .personality

@ .cantunwind directive can't be used with .handlerdata directive nor
@ .personality directive.  This test case check for the diagnostics for
@ the conflicts.


        .syntax unified
        .text

@-------------------------------------------------------------------------------
@ TEST1: cantunwind + personality
@-------------------------------------------------------------------------------
        .globl  func1
        .align  2
        .type   func1,%function
        .fnstart
func1:
        .cantunwind
        .personality    __gxx_personality_v0
@ CHECK: error: .personality can't be used with .cantunwind directive
@ CHECK:        .personality __gxx_personality_v0
@ CHECK:        ^
@ CHECK: note: .cantunwind was specified here
@ CHECK:        .cantunwind
@ CHECK:        ^
        .fnend



@-------------------------------------------------------------------------------
@ TEST2: cantunwind + handlerdata
@-------------------------------------------------------------------------------
        .globl  func2
        .align  2
        .type   func2,%function
        .fnstart
func2:
        .cantunwind
        .handlerdata
@ CHECK: error: .handlerdata can't be used with .cantunwind directive
@ CHECK:        .handlerdata
@ CHECK:        ^
@ CHECK: note: .cantunwind was specified here
@ CHECK:        .cantunwind
@ CHECK:        ^
        .fnend



@-------------------------------------------------------------------------------
@ TEST3: personality + cantunwind
@-------------------------------------------------------------------------------
        .globl  func3
        .align  2
        .type   func3,%function
        .fnstart
func3:
        .personality    __gxx_personality_v0
        .cantunwind
@ CHECK: error: .cantunwind can't be used with .personality directive
@ CHECK:        .cantunwind
@ CHECK:        ^
@ CHECK: note: .personality was specified here
@ CHECK:        .personality __gxx_personality_v0
@ CHECK:        ^
        .fnend



@-------------------------------------------------------------------------------
@ TEST4: handlerdata + cantunwind
@-------------------------------------------------------------------------------
        .globl  func4
        .align  2
        .type   func4,%function
        .fnstart
func4:
        .handlerdata
        .cantunwind
@ CHECK: error: .cantunwind can't be used with .handlerdata directive
@ CHECK:        .cantunwind
@ CHECK:        ^
@ CHECK: note: .handlerdata was specified here
@ CHECK:        .handlerdata
@ CHECK:        ^
        .fnend



@-------------------------------------------------------------------------------
@ TEST5: cantunwind + fnstart
@-------------------------------------------------------------------------------
        .globl  func5
        .align  2
        .type   func5,%function
        .cantunwind
@ CHECK: error: .fnstart must precede .cantunwind directive
@ CHECK:        .cantunwind
@ CHECK:        ^
        .fnstart
func5:
        .fnend
