; REQUIRES: asserts
; RUN: opt -S -inline -stats < %s 2>&1 | FileCheck %s
; CHECK: Number of functions inlined

; RUN: opt -S -inline -functionattrs -stats < %s 2>&1 | FileCheck -check-prefix=CHECK-FUNCTIONATTRS %s
; CHECK-FUNCTIONATTRS: Number of call sites deleted, not inlined

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.8"

define internal i32 @test(i32 %x, i32 %y, i32 %z) nounwind {
entry:
  %0 = add nsw i32 %y, %z                         ; <i32> [#uses=1]
  %1 = mul i32 %0, %x                             ; <i32> [#uses=1]
  %2 = mul i32 %y, %z                             ; <i32> [#uses=1]
  %3 = add nsw i32 %1, %2                         ; <i32> [#uses=1]
  ret i32 %3
}

define i32 @test2() nounwind {
entry:
  %0 = call i32 @test(i32 1, i32 2, i32 4) nounwind ; <i32> [#uses=1]
  ret i32 14
}


