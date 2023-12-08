; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s
; PR3171

; CHECK: count is 2

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

	%struct.Foo = type { i32 }
	%struct.NonPod = type { [2 x %struct.Foo] }

define void @_Z3foov() nounwind {
entry:
	%x = alloca %struct.NonPod, align 8		; <ptr> [#uses=2]
	%0 = getelementptr %struct.NonPod, ptr %x, i32 0, i32 0		; <ptr> [#uses=1]
	%1 = getelementptr [2 x %struct.Foo], ptr %0, i32 1, i32 0		; <ptr> [#uses=1]
	br label %bb1.i

bb1.i:		; preds = %bb2.i, %entry
	%.0.i = phi ptr [ %1, %entry ], [ %4, %bb2.i ]		; <ptr> [#uses=2]
	%2 = getelementptr %struct.NonPod, ptr %x, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	%3 = icmp eq ptr %.0.i, %2		; <i1> [#uses=1]
	br i1 %3, label %_ZN6NonPodD1Ev.exit, label %bb2.i

bb2.i:		; preds = %bb1.i
	%4 = getelementptr %struct.Foo, ptr %.0.i, i32 -1		; <ptr> [#uses=1]
	br label %bb1.i

_ZN6NonPodD1Ev.exit:		; preds = %bb1.i
	ret void
}

