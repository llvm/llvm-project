; RUN: llc -O3 < %s | FileCheck %s 
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "arm64--linux-gnu"

; Function Attrs: nounwind readnone
define i32 @test1(i8 %a) {
; CHECK-LABEL: @test1
; CHECK: ubfx {{w[0-9]+}}, w0, #3, #5
entry:
  %conv = zext i8 %a to i32
  %shr1 = lshr i32 %conv, 3
  ret i32 %shr1
}

; Function Attrs: nounwind readnone
define i32 @test2(i8 %a) {
; CHECK-LABEL: @test2
; CHECK: and {{w[0-9]+}}, w0, #0xff
; CHECK: ubfx {{w[0-9]+}}, w0, #3, #5
entry:
  %conv = zext i8 %a to i32
  %cmp = icmp ugt i8 %a, 47
  %shr5 = lshr i32 %conv, 3
  %retval.0 = select i1 %cmp, i32 %shr5, i32 %conv
  ret i32 %retval.0
}


