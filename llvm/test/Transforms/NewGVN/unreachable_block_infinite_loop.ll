; RUN: opt -passes=newgvn -disable-output < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0"

define i32 @test2() nounwind ssp {
entry:
    ret i32 0

unreachable_block:
    %a = add i32 %a, 1
    ret i32 %a
}

define i32 @pr23096_test0() {
entry:
  br label %bb0

bb1:
  %ptr1 = ptrtoint ptr %ptr2 to i64
  %ptr2 = inttoptr i64 %ptr1 to ptr
  br i1 undef, label %bb0, label %bb1

bb0:
  %phi = phi ptr [ undef, %entry ], [ %ptr2, %bb1 ]
  %load = load i32, ptr %phi
  ret i32 %load
}

define i32 @pr23096_test1() {
entry:
  br label %bb0

bb1:
  %ptr1 = getelementptr i32, ptr %ptr2, i32 0
  %ptr2 = getelementptr i32, ptr %ptr1, i32 0
  br i1 undef, label %bb0, label %bb1

bb0:
  %phi = phi ptr [ undef, %entry ], [ %ptr2, %bb1 ]
  %load = load i32, ptr %phi
  ret i32 %load
}
