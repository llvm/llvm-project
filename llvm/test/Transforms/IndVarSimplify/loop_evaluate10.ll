; RUN: opt < %s -indvars -S \
; RUN:   | grep "%b.1 = phi i32 [ 2, %bb ], [ 1, %bb2 ]"
;
; This loop has multiple exits, and the value of %b1 depends on which
; exit is taken. Indvars should correctly compute the exit values.
;
; XFAIL: *
; Indvars does not currently replace loop invariant values unless all
; loop exits have the same exit value. We could handle some cases,
; such as this, by making getSCEVAtScope() sensitive to a particular
; loop exit.  See PR11388.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-pc-linux-gnu"
	%struct..0anon = type <{ i8, [3 x i8] }>

define i32 @main() nounwind {
entry:
	br label %bb2

bb2:		; preds = %bb, %entry
	%sdata.0 = phi i32 [ 1, %entry ], [ %ins10, %bb ]		; <i32> [#uses=2]
	%b.0 = phi i32 [ 0, %entry ], [ %t0, %bb ]		; <i32> [#uses=2]
	%tmp6 = trunc i32 %sdata.0 to i8		; <i8> [#uses=2]
	%t2 = and i8 %tmp6, 1		; <i8> [#uses=1]
	%t3 = icmp eq i8 %t2, 0		; <i1> [#uses=1]
	%t4 = xor i8 %tmp6, 1		; <i8> [#uses=1]
	%tmp8 = zext i8 %t4 to i32		; <i32> [#uses=1]
	%mask9 = and i32 %sdata.0, -256		; <i32> [#uses=1]
	%ins10 = or i32 %tmp8, %mask9		; <i32> [#uses=1]
	br i1 %t3, label %bb3, label %bb

bb:		; preds = %bb2
	%t0 = add i32 %b.0, 1		; <i32> [#uses=3]
	%t1 = icmp sgt i32 %t0, 100		; <i1> [#uses=1]
	br i1 %t1, label %bb3, label %bb2

bb3:		; preds = %bb, %bb2
	%b.1 = phi i32 [ %t0, %bb ], [ %b.0, %bb2 ]		; <i32> [#uses=1]
	%t5 = icmp eq i32 %b.1, 1		; <i1> [#uses=1]
	br i1 %t5, label %bb5, label %bb4

bb4:		; preds = %bb3
	tail call void @abort() noreturn nounwind
	unreachable

bb5:		; preds = %bb3
	ret i32 0
}

declare void @abort() noreturn nounwind
