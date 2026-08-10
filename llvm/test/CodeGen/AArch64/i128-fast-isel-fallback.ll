; RUN: llc -O0 -fast-isel -mtriple=arm64-apple-ios7.0 -mcpu=generic < %s | FileCheck %s

; Function Attrs: nounwind ssp
define void @test1() {
  %1 = sext i32 0 to i128
  call void  @test2(i128 %1)
  ret void

; The i128 is 0 so the we can test to make sure it is propagated into the x
; registers that make up the i128 pair

; CHECK:  mov  x0, xzr
; CHECK:  mov  x1, x0
; CHECK:  bl  _test2

}

declare void @test2(i128)
