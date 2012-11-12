; RUN: opt < %s -simplifycfg -S | not grep select
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"
module asm ".globl _foo"
module asm "_foo: ret"
module asm ".globl _i"
module asm ".set _i, 0"
@i = extern_weak global i32		; <i32*> [#uses=2]
@j = common global i32 0		; <i32*> [#uses=1]
@ed = common global double 0.000000e+00, align 8		; <double*> [#uses=1]

define i32 @main() nounwind ssp {
entry:
	br label %bb4

bb:		; preds = %bb4
	br i1 icmp ne (i32* @i, i32* null), label %bb1, label %bb2

bb1:		; preds = %bb
	%0 = load i32* @i, align 4		; <i32> [#uses=1]
	br label %bb3

bb2:		; preds = %bb
	br label %bb3

bb3:		; preds = %bb2, %bb1
	%storemerge = phi i32 [ %0, %bb1 ], [ 0, %bb2 ]		; <i32> [#uses=2]
	store i32 %storemerge, i32* @j
	%1 = sitofp i32 %storemerge to double		; <double> [#uses=1]
	%2 = call double @sin(double %1) nounwind readonly		; <double> [#uses=1]
	%3 = fadd double %2, %d.0		; <double> [#uses=1]
	%4 = add i32 %l.0, 1		; <i32> [#uses=1]
	br label %bb4

bb4:		; preds = %bb3, %entry
	%d.0 = phi double [ undef, %entry ], [ %3, %bb3 ]		; <double> [#uses=2]
	%l.0 = phi i32 [ 0, %entry ], [ %4, %bb3 ]		; <i32> [#uses=2]
	%5 = icmp sgt i32 %l.0, 99		; <i1> [#uses=1]
	br i1 %5, label %bb5, label %bb

bb5:		; preds = %bb4
	store double %d.0, double* @ed, align 8
	ret i32 0
}

declare double @sin(double) nounwind readonly
