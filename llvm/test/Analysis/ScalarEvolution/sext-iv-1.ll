; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

; CHECK: -->  (sext i{{.}} {{{.*}},+,{{.*}}}<%bb1> to i64)
; CHECK: -->  (sext i{{.}} {{{.*}},+,{{.*}}}<%bb1> to i64)
; CHECK: -->  (sext i{{.}} {{{.*}},+,{{.*}}}<%bb1> to i64)
; CHECK: -->  (sext i{{.}} {{{.*}},+,{{.*}}}<%bb1> to i64)
; CHECK: -->  (sext i{{.}} {{{.*}},+,{{.*}}}<%bb1> to i64)
; CHECK-NOT: -->  (sext

; Don't convert (sext {...,+,...}) to {sext(...),+,sext(...)} in cases
; where the trip count is not within range.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo0(double* nocapture %x) nounwind {
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i64 [ -128, %bb1.thread ], [ %8, %bb1 ]		; <i64> [#uses=3]
	%0 = trunc i64 %i.0.reg2mem.0 to i7		; <i8> [#uses=1]
	%1 = trunc i64 %i.0.reg2mem.0 to i9		; <i8> [#uses=1]
	%2 = sext i9 %1 to i64		; <i64> [#uses=1]
	%3 = getelementptr double, double* %x, i64 %2		; <double*> [#uses=1]
	%4 = load double, double* %3, align 8		; <double> [#uses=1]
	%5 = fmul double %4, 3.900000e+00		; <double> [#uses=1]
	%6 = sext i7 %0 to i64		; <i64> [#uses=1]
	%7 = getelementptr double, double* %x, i64 %6		; <double*> [#uses=1]
	store double %5, double* %7, align 8
	%8 = add i64 %i.0.reg2mem.0, 1		; <i64> [#uses=2]
	%9 = icmp sgt i64 %8, 127		; <i1> [#uses=1]
	br i1 %9, label %return, label %bb1

return:		; preds = %bb1
	ret void
}

define void @foo1(double* nocapture %x) nounwind {
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i64 [ -128, %bb1.thread ], [ %8, %bb1 ]		; <i64> [#uses=3]
	%0 = trunc i64 %i.0.reg2mem.0 to i8		; <i8> [#uses=1]
	%1 = trunc i64 %i.0.reg2mem.0 to i9		; <i8> [#uses=1]
	%2 = sext i9 %1 to i64		; <i64> [#uses=1]
	%3 = getelementptr double, double* %x, i64 %2		; <double*> [#uses=1]
	%4 = load double, double* %3, align 8		; <double> [#uses=1]
	%5 = fmul double %4, 3.900000e+00		; <double> [#uses=1]
	%6 = sext i8 %0 to i64		; <i64> [#uses=1]
	%7 = getelementptr double, double* %x, i64 %6		; <double*> [#uses=1]
	store double %5, double* %7, align 8
	%8 = add i64 %i.0.reg2mem.0, 1		; <i64> [#uses=2]
	%9 = icmp sgt i64 %8, 128		; <i1> [#uses=1]
	br i1 %9, label %return, label %bb1

return:		; preds = %bb1
	ret void
}

define void @foo2(double* nocapture %x) nounwind {
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i64 [ -129, %bb1.thread ], [ %8, %bb1 ]		; <i64> [#uses=3]
	%0 = trunc i64 %i.0.reg2mem.0 to i8		; <i8> [#uses=1]
	%1 = trunc i64 %i.0.reg2mem.0 to i9		; <i8> [#uses=1]
	%2 = sext i9 %1 to i64		; <i64> [#uses=1]
	%3 = getelementptr double, double* %x, i64 %2		; <double*> [#uses=1]
	%4 = load double, double* %3, align 8		; <double> [#uses=1]
	%5 = fmul double %4, 3.900000e+00		; <double> [#uses=1]
	%6 = sext i8 %0 to i64		; <i64> [#uses=1]
	%7 = getelementptr double, double* %x, i64 %6		; <double*> [#uses=1]
	store double %5, double* %7, align 8
	%8 = add i64 %i.0.reg2mem.0, 1		; <i64> [#uses=2]
	%9 = icmp sgt i64 %8, 127		; <i1> [#uses=1]
	br i1 %9, label %return, label %bb1

return:		; preds = %bb1
	ret void
}

define void @foo3(double* nocapture %x) nounwind {
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i64 [ -128, %bb1.thread ], [ %8, %bb1 ]		; <i64> [#uses=3]
	%0 = trunc i64 %i.0.reg2mem.0 to i8		; <i8> [#uses=1]
	%1 = trunc i64 %i.0.reg2mem.0 to i9		; <i8> [#uses=1]
	%2 = sext i9 %1 to i64		; <i64> [#uses=1]
	%3 = getelementptr double, double* %x, i64 %2		; <double*> [#uses=1]
	%4 = load double, double* %3, align 8		; <double> [#uses=1]
	%5 = fmul double %4, 3.900000e+00		; <double> [#uses=1]
	%6 = sext i8 %0 to i64		; <i64> [#uses=1]
	%7 = getelementptr double, double* %x, i64 %6		; <double*> [#uses=1]
	store double %5, double* %7, align 8
	%8 = add i64 %i.0.reg2mem.0, -1		; <i64> [#uses=2]
	%9 = icmp sgt i64 %8, 127		; <i1> [#uses=1]
	br i1 %9, label %return, label %bb1

return:		; preds = %bb1
	ret void
}
