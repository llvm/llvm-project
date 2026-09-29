; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s
@array = external global [0 x i32]

define i64 @foo(i32 %i) {
; CHECK: foo
; CHECK:  adrp  x[[REG:[0-9]+]], _array@GOTPAGE
; CHECK:  ldr x[[REG1:[0-9]+]], [x[[REG]], _array@GOTPAGEOFF]
; CHECK:  ldrsw x0, [x[[REG1]], w0, sxtw #2]
; CHECK:  ret
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @array, i64 0, i64 %idxprom
  %tmp1 = load i32, ptr %arrayidx, align 4
  %conv = sext i32 %tmp1 to i64
  ret i64 %conv
}
