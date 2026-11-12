; RUN: llc < %s -mtriple=armv6-apple-darwin10
; <rdar://problem/8782198>
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:64-n32"
target triple = "armv6-apple-darwin10"

define void @func() nounwind optsize {
entry:
  %buf = alloca [8096 x i8], align 1
  br label %bb

bb:
  store i8 undef, ptr %buf, align 1
  ret void
}
