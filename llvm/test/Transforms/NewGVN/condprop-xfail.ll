; XFAIL: *
; RUN: opt < %s -basic-aa -newgvn -S | FileCheck %s

@a = external global i32		; <i32*> [#uses=7]

;; NewGVN takes two passes to get this, because we prune predicateinfo
; CHECK-LABEL: @test1(
define i32 @test1() nounwind {
entry:
	%0 = load i32, i32* @a, align 4
	%1 = icmp eq i32 %0, 4
	br i1 %1, label %bb, label %bb1

bb:		; preds = %entry
	br label %bb8

bb1:		; preds = %entry
	%2 = load i32, i32* @a, align 4
	%3 = icmp eq i32 %2, 5
	br i1 %3, label %bb2, label %bb3

bb2:		; preds = %bb1
	br label %bb8

bb3:		; preds = %bb1
	%4 = load i32, i32* @a, align 4
	%5 = icmp eq i32 %4, 4
; CHECK: br i1 false, label %bb4, label %bb5
	br i1 %5, label %bb4, label %bb5

bb4:		; preds = %bb3
	%6 = load i32, i32* @a, align 4
	%7 = add i32 %6, 5
	br label %bb8

bb5:		; preds = %bb3
	%8 = load i32, i32* @a, align 4
	%9 = icmp eq i32 %8, 5
; CHECK: br i1 false, label %bb6, label %bb7
	br i1 %9, label %bb6, label %bb7

bb6:		; preds = %bb5
	%10 = load i32, i32* @a, align 4
	%11 = add i32 %10, 4
	br label %bb8

bb7:		; preds = %bb5
	%12 = load i32, i32* @a, align 4
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb4, %bb2, %bb
	%.0 = phi i32 [ %12, %bb7 ], [ %11, %bb6 ], [ %7, %bb4 ], [ 4, %bb2 ], [ 5, %bb ]
	br label %return

return:		; preds = %bb8
	ret i32 %.0
}
;; NewGVN takes two passes to get test[6,8] and test[6,8]_fp's main part
;; The icmp ne requires an equality table that inserts the inequalities for each
;; discovered equality while processing.
; CHECK-LABEL: @test6(
define i1 @test6(i32 %x, i32 %y) {
  %cmp2 = icmp ne i32 %x, %y
  %cmp = icmp eq i32 %x, %y
  %cmp3 = icmp eq i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test6_fp(
define i1 @test6_fp(float %x, float %y) {
  %cmp2 = fcmp une float %x, %y
  %cmp = fcmp oeq float %x, %y
  %cmp3 = fcmp oeq float  %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}
; CHECK-LABEL: @test8(
define i1 @test8(i32 %x, i32 %y) {
  %cmp2 = icmp sle i32 %x, %y
  %cmp = icmp sgt i32 %x, %y
  %cmp3 = icmp sgt i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test8_fp(
define i1 @test8_fp(float %x, float %y) {
  %cmp2 = fcmp ule float %x, %y
  %cmp = fcmp ogt float %x, %y
  %cmp3 = fcmp ogt float %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}

