; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep phi | count 1
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep mul | count 1
; ModuleID = '<stdin>'
; Make sure examining a fuller expression outside the loop doesn't cause us to create a second
; IV of stride %3.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%struct.anon = type { %struct.obj*, %struct.obj* }
	%struct.obj = type { i16, i16, { %struct.anon } }
@heap_size = external global i32		; <i32*> [#uses=1]
@"\01LC85" = external constant [39 x i8]		; <[39 x i8]*> [#uses=1]

declare i32 @sprintf(i8*, i8*, ...) nounwind

define %struct.obj* @gc_status(%struct.obj* %args) nounwind {
entry:
	br label %bb1.i

bb.i2:		; preds = %bb2.i3
	%indvar.next24 = add i32 %m.0.i, 1		; <i32> [#uses=1]
	br label %bb1.i

bb1.i:		; preds = %bb.i2, %entry
	%m.0.i = phi i32 [ 0, %entry ], [ %indvar.next24, %bb.i2 ]		; <i32> [#uses=4]
	%0 = icmp slt i32 %m.0.i, 0		; <i1> [#uses=1]
	br i1 %0, label %bb2.i3, label %nactive_heaps.exit

bb2.i3:		; preds = %bb1.i
	%1 = load %struct.obj** null, align 4		; <%struct.obj*> [#uses=1]
	%2 = icmp eq %struct.obj* %1, null		; <i1> [#uses=1]
	br i1 %2, label %nactive_heaps.exit, label %bb.i2

nactive_heaps.exit:		; preds = %bb2.i3, %bb1.i
	%3 = load i32* @heap_size, align 4		; <i32> [#uses=1]
	%4 = mul i32 %3, %m.0.i		; <i32> [#uses=1]
	%5 = sub i32 %4, 0		; <i32> [#uses=1]
	%6 = tail call i32 (i8*, i8*, ...)* @sprintf(i8* null, i8* getelementptr ([39 x i8]* @"\01LC85", i32 0, i32 0), i32 %m.0.i, i32 0, i32 %5, i32 0) nounwind		; <i32> [#uses=0]
	ret %struct.obj* null
}
