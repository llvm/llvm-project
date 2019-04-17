; XFAIL: *
;; NewGVN zaps the strlens, but currently takes two iterations to evaluate the conditions, because
;; we prune predicateinfo, and the icmps only become equivalent after the strlens are zapped
; Two occurrences of strlen should be zapped.
; RUN: opt < %s -basicaa -newgvn -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"

define i32 @test(i32 %g, i8* %P) nounwind  {
entry:
	%tmp2 = call i32 @strlen( i8* %P ) nounwind readonly 		; <i32> [#uses=1]
	%tmp3 = icmp eq i32 %tmp2, 100		; <i1> [#uses=1]
	%tmp34 = zext i1 %tmp3 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp34, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb6

bb:		; preds = %entry
	br label %bb27

bb6:		; preds = %entry
	%tmp8 = add i32 %g, 42		; <i32> [#uses=2]
	%tmp10 = call i32 @strlen( i8* %P ) nounwind readonly 		; <i32> [#uses=1]
	%tmp11 = icmp eq i32 %tmp10, 100		; <i1> [#uses=1]
	%tmp1112 = zext i1 %tmp11 to i8		; <i8> [#uses=1]
	%toBool13 = icmp ne i8 %tmp1112, 0		; <i1> [#uses=1]
	br i1 %toBool13, label %bb14, label %bb16

bb14:		; preds = %bb6
	br label %bb27

bb16:		; preds = %bb6
	%tmp18 = mul i32 %tmp8, 2		; <i32> [#uses=1]
	%tmp20 = call i32 @strlen( i8* %P ) nounwind readonly 		; <i32> [#uses=1]
	%tmp21 = icmp eq i32 %tmp20, 100		; <i1> [#uses=1]
	%tmp2122 = zext i1 %tmp21 to i8		; <i8> [#uses=1]
	%toBool23 = icmp ne i8 %tmp2122, 0		; <i1> [#uses=1]
	br i1 %toBool23, label %bb24, label %bb26

bb24:		; preds = %bb16
	br label %bb27

bb26:		; preds = %bb16
	br label %bb27

bb27:		; preds = %bb26, %bb24, %bb14, %bb
	%tmp.0 = phi i32 [ 11, %bb26 ], [ %tmp18, %bb24 ], [ %tmp8, %bb14 ], [ %g, %bb ]		; <i32> [#uses=1]
	br label %return

return:		; preds = %bb27
	ret i32 %tmp.0
}

; CHECK: define i32 @test(i32 %g, i8* %P) #0 {
; CHECK: entry:
; CHECK:   %tmp2 = call i32 @strlen(i8* %P) #1
; CHECK:   %tmp3 = icmp eq i32 %tmp2, 100
; CHECK:   %tmp34 = zext i1 %tmp3 to i8
; CHECK:   br i1 %tmp3, label %bb, label %bb6
; CHECK: bb:
; CHECK:   br label %bb27
; CHECK: bb6:
; CHECK:   %tmp8 = add i32 %g, 42
; CHECK:   br i1 false, label %bb14, label %bb16
; CHECK: bb14:
; CHECK:   br label %bb27
; CHECK: bb16:
; CHECK:   %tmp18 = mul i32 %tmp8, 2
; CHECK:   br i1 false, label %bb24, label %bb26
; CHECK: bb24:
; CHECK:   br label %bb27
; CHECK: bb26:
; CHECK:   br label %bb27
; CHECK: bb27:
; CHECK:   %tmp.0 = phi i32 [ 11, %bb26 ], [ undef, %bb24 ], [ undef, %bb14 ], [ %g, %bb ]
; CHECK:   ret i32 %tmp.0
; CHECK: }

declare i32 @strlen(i8*) nounwind readonly 
