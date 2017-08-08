; REQUIRES: asserts
; RUN: llc < %s -march=x86-64 -mattr=+sse3,+sse4.1 -mcpu=penryn -stats 2>&1 | grep "9 machinelicm"
; RUN: llc < %s -march=x86-64 -mattr=+sse3,+sse4.1 -mcpu=penryn | FileCheck %s
; rdar://6627786
; rdar://7792037

target triple = "x86_64-apple-darwin10.0"
	%struct.Key = type { i64 }
	%struct.__Rec = type opaque
	%struct.__vv = type {  }

define %struct.__vv* @t(%struct.Key* %desc, i64 %p) nounwind ssp {
entry:
	br label %bb4

bb4:		; preds = %bb.i, %bb26, %bb4, %entry
; CHECK: %bb4
; CHECK: xorl
; CHECK: callq
; CHECK: xorl
; CHECK: xorl
; CHECK: movq

	%0 = call i32 (...) @xxGetOffsetForCode(i32 undef) nounwind		; <i32> [#uses=0]
	%ins = or i64 %p, 2097152		; <i64> [#uses=1]
	%1 = call i32 (...) @xxCalculateMidType(%struct.Key* %desc, i32 0) nounwind		; <i32> [#uses=1]
	%cond = icmp eq i32 %1, 1		; <i1> [#uses=1]
	br i1 %cond, label %bb26, label %bb4

bb26:		; preds = %bb4
	%2 = and i64 %ins, 15728640		; <i64> [#uses=1]
	%cond.i = icmp eq i64 %2, 1048576		; <i1> [#uses=1]
	br i1 %cond.i, label %bb.i, label %bb4

bb.i:		; preds = %bb26
	%3 = load i32, i32* null, align 4		; <i32> [#uses=1]
	%4 = uitofp i32 %3 to float		; <float> [#uses=1]
	%.sum13.i = add i64 0, 4		; <i64> [#uses=1]
	%5 = getelementptr i8, i8* null, i64 %.sum13.i		; <i8*> [#uses=1]
	%6 = bitcast i8* %5 to i32*		; <i32*> [#uses=1]
	%7 = load i32, i32* %6, align 4		; <i32> [#uses=1]
	%8 = uitofp i32 %7 to float		; <float> [#uses=1]
	%.sum.i = add i64 0, 8		; <i64> [#uses=1]
	%9 = getelementptr i8, i8* null, i64 %.sum.i		; <i8*> [#uses=1]
	%10 = bitcast i8* %9 to i32*		; <i32*> [#uses=1]
	%11 = load i32, i32* %10, align 4		; <i32> [#uses=1]
	%12 = uitofp i32 %11 to float		; <float> [#uses=1]
	%13 = insertelement <4 x float> undef, float %4, i32 0		; <<4 x float>> [#uses=1]
	%14 = insertelement <4 x float> %13, float %8, i32 1		; <<4 x float>> [#uses=1]
	%15 = insertelement <4 x float> %14, float %12, i32 2		; <<4 x float>> [#uses=1]
	store <4 x float> %15, <4 x float>* null, align 16
	br label %bb4
}

declare i32 @xxGetOffsetForCode(...)

declare i32 @xxCalculateMidType(...)
