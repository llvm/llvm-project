; RUN: llc < %s -mtriple=lanai-unknown-unknown | FileCheck %s

; Test right-shift i64 lowering does not result in call being inserted.

; CHECK-LABEL: shift
; CHECK-NOT: bt __lshrdi3
; CHECK: %rv
define i64 @shift(i64 inreg, i32 inreg) {
  %3 = zext i32 %1 to i64
  %4 = lshr i64 %0, %3
  ret i64 %4
}
