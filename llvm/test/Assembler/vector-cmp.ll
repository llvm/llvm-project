; RUN: llvm-as  < %s | llvm-dis  | llvm-as  | llvm-dis  | FileCheck %s
; RUN: verify-uselistorder  %s
; PR2317
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9.2.2"

; CHECK: @1 = global <4 x i1> <i1 icmp slt (i32 ptrtoint (ptr @B to i32), i32 1), i1 true, i1 false, i1 true>

define <4 x i1> @foo(<4 x float> %a, <4 x float> %b) nounwind  {
entry:
  %cmp = fcmp olt <4 x float> %a, %b		; <4 x i32> [#uses=1]
  ret <4 x i1> %cmp
}

@0 = global <4 x i1> icmp slt ( <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32>  <i32 1, i32 2, i32 1, i32 2> )
@B = external global i32
@1 = global <4 x i1> icmp slt ( <4 x i32> <i32 ptrtoint (ptr @B to i32), i32 1, i32 1, i32 1>, <4 x i32>  <i32 1, i32 2, i32 1, i32 2> )
