; RUN: llc < %s -march=x86
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define hidden float @__floatundisf(i64 %u) nounwind readnone {
entry:
	%0 = icmp ugt i64 %u, 9007199254740991		; <i1> [#uses=1]
	br i1 %0, label %bb, label %bb2

bb:		; preds = %entry
	%1 = and i64 %u, 2047		; <i64> [#uses=1]
	%2 = icmp eq i64 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb2, label %bb1

bb1:		; preds = %bb
	%3 = or i64 %u, 2048		; <i64> [#uses=1]
	%4 = and i64 %3, -2048		; <i64> [#uses=1]
	br label %bb2

bb2:		; preds = %bb1, %bb, %entry
	%u_addr.0 = phi i64 [ %4, %bb1 ], [ %u, %entry ], [ %u, %bb ]		; <i64> [#uses=2]
	%5 = lshr i64 %u_addr.0, 32		; <i64> [#uses=1]
	%6 = trunc i64 %5 to i32		; <i32> [#uses=1]
	%7 = uitofp i32 %6 to double		; <double> [#uses=1]
	%8 = fmul double %7, 0x41F0000000000000		; <double> [#uses=1]
	%9 = trunc i64 %u_addr.0 to i32		; <i32> [#uses=1]
	%10 = uitofp i32 %9 to double		; <double> [#uses=1]
	%11 = fadd double %10, %8		; <double> [#uses=1]
	%12 = fptrunc double %11 to float		; <float> [#uses=1]
	ret float %12
}
