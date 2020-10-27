; RUN: opt < %s -scalar-evolution -analyze -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

; Convert (sext {-128,+,1}) to {sext(-128),+,sext(1)}, since the
; trip count is within range where this is safe.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(double* nocapture %x) nounwind {
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i64 [ -128, %bb1.thread ], [ %8, %bb1 ]		; <i64> [#uses=3]
; CHECK: %i.0.reg2mem.0
; CHECK-NEXT: -->  {-128,+,1}<%bb1>{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 127
	%0 = trunc i64 %i.0.reg2mem.0 to i8		; <i8> [#uses=1]
; CHECK: %0
; CHECK-NEXT: -->  {-128,+,1}<%bb1>{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 127
	%1 = trunc i64 %i.0.reg2mem.0 to i9		; <i8> [#uses=1]
; CHECK: %1
; CHECK-NEXT: -->  {-128,+,1}<%bb1>{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 127
	%2 = sext i9 %1 to i64		; <i64> [#uses=1]
; CHECK: %2
; CHECK-NEXT: -->  {-128,+,1}<nsw><%bb1>{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 127
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
