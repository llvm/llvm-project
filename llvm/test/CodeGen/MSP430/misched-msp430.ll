; RUN: llc < %s -mtriple=msp430-unknown-unknown -enable-misched | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"

@y = common global i16 0, align 2
@x = common global i16 0, align 2

; Test that the MI Scheduler's initPolicy does not crash when i32 is
; unsupported. The content of the asm check below is unimportant. It
; only verifies that the code generator ran successfully.
;
; CHECK-LABEL: @f
; CHECK: mov.w &y, &x
; CHECK: ret
define void @f() {
entry:
  %0 = load i16, i16* @y, align 2
  store i16 %0, i16* @x, align 2
  ret void
}
