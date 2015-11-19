; RUN: opt < %s -analyze -scalar-evolution -S | FileCheck %s

; Indvars should be able to find the trip count for the bb6 loop
; without using a maximum calculation (icmp, select) because it should
; be able to prove that the comparison is guarded by an appropriate
; conditional branch. Unfortunately, indvars is not yet able to find
; the comparison for the other two loops in this testcase.
;
; CHECK: Loop %bb6: backedge-taken count is (-1 + %w)

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"

define void @foo(i8* %r, i32 %s, i32 %w, i32 %x, i8* %j, i32 %d) nounwind {
entry:
	%0 = mul i32 %x, %w		; <i32> [#uses=2]
	%1 = mul i32 %x, %w		; <i32> [#uses=1]
	%2 = sdiv i32 %1, 4		; <i32> [#uses=1]
	%.sum2 = add i32 %2, %0		; <i32> [#uses=2]
	%cond = icmp eq i32 %d, 1		; <i1> [#uses=1]
	br i1 %cond, label %bb29, label %bb10.preheader

bb10.preheader:		; preds = %entry
	%3 = icmp sgt i32 %x, 0		; <i1> [#uses=1]
	br i1 %3, label %bb.nph9, label %bb18.loopexit

bb.nph7:		; preds = %bb7.preheader
	%4 = mul i32 %y.08, %w		; <i32> [#uses=1]
	%5 = mul i32 %y.08, %s		; <i32> [#uses=1]
	%6 = add i32 %5, 1		; <i32> [#uses=1]
	br label %bb6

bb6:		; preds = %bb7, %bb.nph7
	%x.06 = phi i32 [ %13, %bb7 ], [ 0, %bb.nph7 ]		; <i32> [#uses=3]
	%7 = add i32 %x.06, %4		; <i32> [#uses=1]
	%8 = shl i32 %x.06, 1		; <i32> [#uses=1]
	%9 = add i32 %6, %8		; <i32> [#uses=1]
	%10 = getelementptr i8, i8* %r, i32 %9		; <i8*> [#uses=1]
	%11 = load i8, i8* %10, align 1		; <i8> [#uses=1]
	%12 = getelementptr i8, i8* %j, i32 %7		; <i8*> [#uses=1]
	store i8 %11, i8* %12, align 1
	%13 = add i32 %x.06, 1		; <i32> [#uses=2]
	br label %bb7

bb7:		; preds = %bb6
	%14 = icmp slt i32 %13, %w		; <i1> [#uses=1]
	br i1 %14, label %bb6, label %bb7.bb9_crit_edge

bb7.bb9_crit_edge:		; preds = %bb7
	br label %bb9

bb9:		; preds = %bb7.preheader, %bb7.bb9_crit_edge
	%15 = add i32 %y.08, 1		; <i32> [#uses=2]
	br label %bb10

bb10:		; preds = %bb9
	%16 = icmp slt i32 %15, %x		; <i1> [#uses=1]
	br i1 %16, label %bb7.preheader, label %bb10.bb18.loopexit_crit_edge

bb10.bb18.loopexit_crit_edge:		; preds = %bb10
	br label %bb10.bb18.loopexit_crit_edge.split

bb10.bb18.loopexit_crit_edge.split:		; preds = %bb.nph9, %bb10.bb18.loopexit_crit_edge
	br label %bb18.loopexit

bb.nph9:		; preds = %bb10.preheader
	%17 = icmp sgt i32 %w, 0		; <i1> [#uses=1]
	br i1 %17, label %bb.nph9.split, label %bb10.bb18.loopexit_crit_edge.split

bb.nph9.split:		; preds = %bb.nph9
	br label %bb7.preheader

bb7.preheader:		; preds = %bb.nph9.split, %bb10
	%y.08 = phi i32 [ %15, %bb10 ], [ 0, %bb.nph9.split ]		; <i32> [#uses=3]
	br i1 true, label %bb.nph7, label %bb9

bb.nph5:		; preds = %bb18.loopexit
	%18 = sdiv i32 %w, 2		; <i32> [#uses=1]
	%19 = icmp slt i32 %w, 2		; <i1> [#uses=1]
	%20 = sdiv i32 %x, 2		; <i32> [#uses=1]
	br i1 %19, label %bb18.bb20_crit_edge.split, label %bb.nph5.split

bb.nph5.split:		; preds = %bb.nph5
	br label %bb13

bb13:		; preds = %bb18, %bb.nph5.split
	%y.14 = phi i32 [ %42, %bb18 ], [ 0, %bb.nph5.split ]		; <i32> [#uses=4]
	%21 = mul i32 %18, %y.14		; <i32> [#uses=2]
	%22 = shl i32 %y.14, 1		; <i32> [#uses=1]
	%23 = srem i32 %y.14, 2		; <i32> [#uses=1]
	%24 = add i32 %23, %22		; <i32> [#uses=1]
	%25 = mul i32 %24, %s		; <i32> [#uses=2]
	br i1 true, label %bb.nph3, label %bb17

bb.nph3:		; preds = %bb13
	%26 = add i32 %21, %0		; <i32> [#uses=1]
	%27 = add i32 %21, %.sum2		; <i32> [#uses=1]
	%28 = sdiv i32 %w, 2		; <i32> [#uses=1]
	br label %bb14

bb14:		; preds = %bb15, %bb.nph3
	%x.12 = phi i32 [ %40, %bb15 ], [ 0, %bb.nph3 ]		; <i32> [#uses=5]
	%29 = shl i32 %x.12, 2		; <i32> [#uses=1]
	%30 = add i32 %29, %25		; <i32> [#uses=1]
	%31 = getelementptr i8, i8* %r, i32 %30		; <i8*> [#uses=1]
	%32 = load i8, i8* %31, align 1		; <i8> [#uses=1]
	%.sum = add i32 %26, %x.12		; <i32> [#uses=1]
	%33 = getelementptr i8, i8* %j, i32 %.sum		; <i8*> [#uses=1]
	store i8 %32, i8* %33, align 1
	%34 = shl i32 %x.12, 2		; <i32> [#uses=1]
	%35 = or i32 %34, 2		; <i32> [#uses=1]
	%36 = add i32 %35, %25		; <i32> [#uses=1]
	%37 = getelementptr i8, i8* %r, i32 %36		; <i8*> [#uses=1]
	%38 = load i8, i8* %37, align 1		; <i8> [#uses=1]
	%.sum6 = add i32 %27, %x.12		; <i32> [#uses=1]
	%39 = getelementptr i8, i8* %j, i32 %.sum6		; <i8*> [#uses=1]
	store i8 %38, i8* %39, align 1
	%40 = add i32 %x.12, 1		; <i32> [#uses=2]
	br label %bb15

bb15:		; preds = %bb14
	%41 = icmp sgt i32 %28, %40		; <i1> [#uses=1]
	br i1 %41, label %bb14, label %bb15.bb17_crit_edge

bb15.bb17_crit_edge:		; preds = %bb15
	br label %bb17

bb17:		; preds = %bb15.bb17_crit_edge, %bb13
	%42 = add i32 %y.14, 1		; <i32> [#uses=2]
	br label %bb18

bb18.loopexit:		; preds = %bb10.bb18.loopexit_crit_edge.split, %bb10.preheader
	%43 = icmp slt i32 %x, 2		; <i1> [#uses=1]
	br i1 %43, label %bb20, label %bb.nph5

bb18:		; preds = %bb17
	%44 = icmp sgt i32 %20, %42		; <i1> [#uses=1]
	br i1 %44, label %bb13, label %bb18.bb20_crit_edge

bb18.bb20_crit_edge:		; preds = %bb18
	br label %bb18.bb20_crit_edge.split

bb18.bb20_crit_edge.split:		; preds = %bb18.bb20_crit_edge, %bb.nph5
	br label %bb20

bb20:		; preds = %bb18.bb20_crit_edge.split, %bb18.loopexit
	switch i32 %d, label %return [
		i32 3, label %bb22
		i32 1, label %bb29
	]

bb22:		; preds = %bb20
	%45 = mul i32 %x, %w		; <i32> [#uses=1]
	%46 = sdiv i32 %45, 4		; <i32> [#uses=1]
	%.sum3 = add i32 %46, %.sum2		; <i32> [#uses=2]
	%47 = add i32 %x, 15		; <i32> [#uses=1]
	%48 = and i32 %47, -16		; <i32> [#uses=1]
	%49 = add i32 %w, 15		; <i32> [#uses=1]
	%50 = and i32 %49, -16		; <i32> [#uses=1]
	%51 = mul i32 %48, %s		; <i32> [#uses=1]
	%52 = icmp sgt i32 %x, 0		; <i1> [#uses=1]
	br i1 %52, label %bb.nph, label %bb26

bb.nph:		; preds = %bb22
	br label %bb23

bb23:		; preds = %bb24, %bb.nph
	%y.21 = phi i32 [ %57, %bb24 ], [ 0, %bb.nph ]		; <i32> [#uses=3]
	%53 = mul i32 %y.21, %50		; <i32> [#uses=1]
	%.sum1 = add i32 %53, %51		; <i32> [#uses=1]
	%54 = getelementptr i8, i8* %r, i32 %.sum1		; <i8*> [#uses=1]
	%55 = mul i32 %y.21, %w		; <i32> [#uses=1]
	%.sum5 = add i32 %55, %.sum3		; <i32> [#uses=1]
	%56 = getelementptr i8, i8* %j, i32 %.sum5		; <i8*> [#uses=1]
	tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %56, i8* %54, i32 %w, i1 false)
	%57 = add i32 %y.21, 1		; <i32> [#uses=2]
	br label %bb24

bb24:		; preds = %bb23
	%58 = icmp slt i32 %57, %x		; <i1> [#uses=1]
	br i1 %58, label %bb23, label %bb24.bb26_crit_edge

bb24.bb26_crit_edge:		; preds = %bb24
	br label %bb26

bb26:		; preds = %bb24.bb26_crit_edge, %bb22
	%59 = mul i32 %x, %w		; <i32> [#uses=1]
	%.sum4 = add i32 %.sum3, %59		; <i32> [#uses=1]
	%60 = getelementptr i8, i8* %j, i32 %.sum4		; <i8*> [#uses=1]
	%61 = mul i32 %x, %w		; <i32> [#uses=1]
	%62 = sdiv i32 %61, 2		; <i32> [#uses=1]
	tail call void @llvm.memset.p0i8.i32(i8* %60, i8 -128, i32 %62, i1 false)
	ret void

bb29:		; preds = %bb20, %entry
	%63 = add i32 %w, 15		; <i32> [#uses=1]
	%64 = and i32 %63, -16		; <i32> [#uses=1]
	%65 = icmp sgt i32 %x, 0		; <i1> [#uses=1]
	br i1 %65, label %bb.nph11, label %bb33

bb.nph11:		; preds = %bb29
	br label %bb30

bb30:		; preds = %bb31, %bb.nph11
	%y.310 = phi i32 [ %70, %bb31 ], [ 0, %bb.nph11 ]		; <i32> [#uses=3]
	%66 = mul i32 %y.310, %64		; <i32> [#uses=1]
	%67 = getelementptr i8, i8* %r, i32 %66		; <i8*> [#uses=1]
	%68 = mul i32 %y.310, %w		; <i32> [#uses=1]
	%69 = getelementptr i8, i8* %j, i32 %68		; <i8*> [#uses=1]
	tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %69, i8* %67, i32 %w, i1 false)
	%70 = add i32 %y.310, 1		; <i32> [#uses=2]
	br label %bb31

bb31:		; preds = %bb30
	%71 = icmp slt i32 %70, %x		; <i1> [#uses=1]
	br i1 %71, label %bb30, label %bb31.bb33_crit_edge

bb31.bb33_crit_edge:		; preds = %bb31
	br label %bb33

bb33:		; preds = %bb31.bb33_crit_edge, %bb29
	%72 = mul i32 %x, %w		; <i32> [#uses=1]
	%73 = getelementptr i8, i8* %j, i32 %72		; <i8*> [#uses=1]
	%74 = mul i32 %x, %w		; <i32> [#uses=1]
	%75 = sdiv i32 %74, 2		; <i32> [#uses=1]
	tail call void @llvm.memset.p0i8.i32(i8* %73, i8 -128, i32 %75, i1 false)
	ret void

return:		; preds = %bb20
	ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind
