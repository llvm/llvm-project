; RUN: opt < %s -indvars -disable-output -stats -info-output-file - | FileCheck %s
; Check that IndVarSimplify is not creating unnecessary canonical IVs
; that will never be used.
; CHECK-NOT: indvars

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@ue = external global i64

define i32 @foo() nounwind {
entry:
	br label %bb38.i

bb14.i27:
	%t0 = load i64* @ue, align 8
	%t1 = sub i64 %t0, %i.0.i35
	%t2 = add i64 %t1, 1
	br i1 undef, label %bb15.i28, label %bb19.i31

bb15.i28:
	br label %bb19.i31

bb19.i31:
	%y.0.i = phi i64 [ %t2, %bb15.i28 ], [ %t2, %bb14.i27 ]
	br label %bb35.i

bb35.i:
	br i1 undef, label %bb37.i, label %bb14.i27

bb37.i:
	%t3 = add i64 %i.0.i35, 1
	br label %bb38.i

bb38.i:
	%i.0.i35 = phi i64 [ 1, %entry ], [ %t3, %bb37.i ]
	br label %bb35.i
}
