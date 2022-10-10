; RUN: opt -aa-pipeline=basic-aa -passes='devirt<1>(inline,function(gvn))' -S < %s | FileCheck %s
; rdar://6295824 and PR6724

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define i32 @foo(i32 ()** noalias nocapture %p, i64* noalias nocapture %q) nounwind ssp {
entry:
  store i32 ()* @bar, i32 ()** %p
  store i64 0, i64* %q
  %tmp3 = load i32 ()*, i32 ()** %p                        ; <i32 ()*> [#uses=1]
  %call = tail call i32 %tmp3() nounwind          ; <i32> [#uses=1]
  ret i32 %call
}
; CHECK-LABEL: @foo(
; CHECK: ret i32 7
; CHECK-LABEL: @bar(
; CHECK: ret i32 7

define internal i32 @bar() nounwind readnone ssp {
entry:
  ret i32 7
}
