; RUN: not llc -mtriple=riscv32 -mattr=+reserve-x1 < %s 2>&1 | FileCheck %s -check-prefix=X1
; RUN: not llc -mtriple=riscv64 -mattr=+reserve-x1 < %s 2>&1 | FileCheck %s -check-prefix=X1
; RUN: not llc -mtriple=riscv32 -mattr=+reserve-x2 < %s 2>&1 | FileCheck %s -check-prefix=X2
; RUN: not llc -mtriple=riscv64 -mattr=+reserve-x2 < %s 2>&1 | FileCheck %s -check-prefix=X2
; RUN: not llc -mtriple=riscv32 -mattr=+reserve-x8 < %s 2>&1 | FileCheck %s -check-prefix=X8
; RUN: not llc -mtriple=riscv64 -mattr=+reserve-x8 < %s 2>&1 | FileCheck %s -check-prefix=X8
; RUN: not llc -mtriple=riscv32 -mattr=+reserve-x10 < %s 2>&1 | FileCheck %s -check-prefix=X10
; RUN: not llc -mtriple=riscv64 -mattr=+reserve-x10 < %s 2>&1 | FileCheck %s -check-prefix=X10
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x11 < %s 2>&1 | FileCheck %s -check-prefix=X11
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x11 < %s 2>&1 | FileCheck %s -check-prefix=X11
; RUN: llc -mtriple=riscv32 <%s
; RUN: llc -mtriple=riscv64 <%s

; This tests combinations when we would expect an error to be produced because
; a reserved register is required by the default ABI. The final test checks no
; errors are produced when no registers are reserved.

define i32 @caller(i32 %a) #0 {
; X1: in function caller {{.*}} Return address register required, but has been reserved.
; X2: in function caller {{.*}} Stack pointer required, but has been reserved.
; X8: in function caller {{.*}} Frame pointer required, but has been reserved.
; X10: in function caller {{.*}} Argument register required, but has been reserved.
; X10: in function caller {{.*}} Return value register required, but has been reserved.
  %call = call i32 @callee(i32 0)
  ret i32 %call
}

declare i32 @callee(i32 %a)

define void @clobber() {
; X11: warning: inline asm clobber list contains reserved registers: X11
  call void asm sideeffect "nop", "~{x11}"()
  ret void
}

attributes #0 = { "frame-pointer"="all" }
