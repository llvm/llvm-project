; RUN: llc < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s
; rdar://9618644

@G = external global i32

define i32 @test(i32 %off) nounwind {
; CHECK-LABEL: test:
; CHECK: adrp x[[REG:[0-9]+]], _G@GOTPAGE
; CHECK: ldr  x[[REG2:[0-9]+]], [x[[REG]], _G@GOTPAGEOFF]
; CHECK: add w0, w[[REG2]], w0
  %tmp = ptrtoint ptr @G to i32
  %tmp1 = add i32 %tmp, %off
  ret i32 %tmp1
}
