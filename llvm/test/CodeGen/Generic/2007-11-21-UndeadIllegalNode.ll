; RUN: llvm-as < %s | llc -o -

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.RETURN = type { i32, i32 }
	%struct.ada__finalization__controlled = type { %struct.system__finalization_root__root_controlled }
	%struct.ada__streams__root_stream_type = type { %struct.ada__tags__dispatch_table* }
	%struct.ada__strings__unbounded__string_access = type { i8*, %struct.RETURN* }
	%struct.ada__strings__unbounded__unbounded_string = type { %struct.ada__finalization__controlled, %struct.ada__strings__unbounded__string_access, i32 }
	%struct.ada__tags__dispatch_table = type { [1 x i32] }
	%struct.exception = type { i8, i8, i32, i8*, i8*, i32, i8* }
	%struct.system__finalization_root__root_controlled = type { %struct.ada__streams__root_stream_type, %struct.system__finalization_root__root_controlled*, %struct.system__finalization_root__root_controlled* }
	%struct.system__standard_library__exception_data = type { i8, i8, i32, i32, %struct.system__standard_library__exception_data*, i32, void ()* }
@C.495.7639 = internal constant %struct.RETURN { i32 1, i32 16 }		; <%struct.RETURN*> [#uses=1]
@ada__strings__index_error = external global %struct.exception		; <%struct.exception*> [#uses=1]
@.str5 = internal constant [16 x i8] c"a-strunb.adb:690"		; <[16 x i8]*> [#uses=1]

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @ada__strings__unbounded__realloc_for_chunk(%struct.ada__strings__unbounded__unbounded_string*, i32)

declare void @__gnat_raise_exception(%struct.system__standard_library__exception_data*, i64)

define void @ada__strings__unbounded__insert__2(%struct.ada__strings__unbounded__unbounded_string* %source, i32 %before, i64 %new_item.0.0) {
entry:
	%tmp24636 = lshr i64 %new_item.0.0, 32		; <i64> [#uses=1]
	%tmp24637 = trunc i64 %tmp24636 to i32		; <i32> [#uses=1]
	%tmp24638 = inttoptr i32 %tmp24637 to %struct.RETURN*		; <%struct.RETURN*> [#uses=2]
	%tmp25 = getelementptr %struct.RETURN* %tmp24638, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp26 = load i32* %tmp25, align 4		; <i32> [#uses=1]
	%tmp29 = getelementptr %struct.RETURN* %tmp24638, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp30 = load i32* %tmp29, align 4		; <i32> [#uses=1]
	%tmp63 = getelementptr %struct.ada__strings__unbounded__unbounded_string* %source, i32 0, i32 1, i32 1		; <%struct.RETURN**> [#uses=5]
	%tmp64 = load %struct.RETURN** %tmp63, align 4		; <%struct.RETURN*> [#uses=1]
	%tmp65 = getelementptr %struct.RETURN* %tmp64, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp66 = load i32* %tmp65, align 4		; <i32> [#uses=1]
	%tmp67 = icmp sgt i32 %tmp66, %before		; <i1> [#uses=1]
	br i1 %tmp67, label %bb77, label %bb

bb:		; preds = %entry
	%tmp71 = getelementptr %struct.ada__strings__unbounded__unbounded_string* %source, i32 0, i32 2		; <i32*> [#uses=4]
	%tmp72 = load i32* %tmp71, align 4		; <i32> [#uses=1]
	%tmp73 = add i32 %tmp72, 1		; <i32> [#uses=1]
	%tmp74 = icmp slt i32 %tmp73, %before		; <i1> [#uses=1]
	br i1 %tmp74, label %bb77, label %bb84

bb77:		; preds = %bb, %entry
	tail call void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @ada__strings__index_error to %struct.system__standard_library__exception_data*), i64 or (i64 zext (i32 ptrtoint ([16 x i8]* @.str5 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.RETURN* @C.495.7639 to i32) to i64), i64 32)) )
	unreachable

bb84:		; preds = %bb
	%tmp93 = sub i32 %tmp30, %tmp26		; <i32> [#uses=2]
	%tmp9394 = sext i32 %tmp93 to i36		; <i36> [#uses=1]
	%tmp95 = shl i36 %tmp9394, 3		; <i36> [#uses=1]
	%tmp96 = add i36 %tmp95, 8		; <i36> [#uses=2]
	%tmp97 = icmp sgt i36 %tmp96, -1		; <i1> [#uses=1]
	%tmp100 = select i1 %tmp97, i36 %tmp96, i36 0		; <i36> [#uses=2]
	%tmp101 = icmp slt i36 %tmp100, 17179869177		; <i1> [#uses=1]
	%tmp100.cast = trunc i36 %tmp100 to i32		; <i32> [#uses=1]
	%min102 = select i1 %tmp101, i32 %tmp100.cast, i32 -8		; <i32> [#uses=1]
	tail call void @ada__strings__unbounded__realloc_for_chunk( %struct.ada__strings__unbounded__unbounded_string* %source, i32 %min102 )
	%tmp148 = load i32* %tmp71, align 4		; <i32> [#uses=4]
	%tmp152 = add i32 %tmp93, 1		; <i32> [#uses=2]
	%tmp153 = icmp sgt i32 %tmp152, -1		; <i1> [#uses=1]
	%max154 = select i1 %tmp153, i32 %tmp152, i32 0		; <i32> [#uses=5]
	%tmp155 = add i32 %tmp148, %max154		; <i32> [#uses=5]
	%tmp315 = getelementptr %struct.ada__strings__unbounded__unbounded_string* %source, i32 0, i32 1, i32 0		; <i8**> [#uses=4]
	%tmp328 = load %struct.RETURN** %tmp63, align 4		; <%struct.RETURN*> [#uses=1]
	%tmp329 = getelementptr %struct.RETURN* %tmp328, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp330 = load i32* %tmp329, align 4		; <i32> [#uses=4]
	%tmp324 = add i32 %max154, %before		; <i32> [#uses=3]
	%tmp331 = sub i32 %tmp324, %tmp330		; <i32> [#uses=1]
	%tmp349 = sub i32 %before, %tmp330		; <i32> [#uses=1]
	%tmp356 = icmp sgt i32 %tmp331, %tmp349		; <i1> [#uses=1]
	%tmp431 = icmp sgt i32 %tmp324, %tmp155		; <i1> [#uses=2]
	br i1 %tmp356, label %bb420, label %bb359

bb359:		; preds = %bb84
	br i1 %tmp431, label %bb481, label %bb382

bb382:		; preds = %bb382, %bb359
	%indvar = phi i32 [ 0, %bb359 ], [ %indvar.next, %bb382 ]		; <i32> [#uses=2]
	%max379.pn = phi i32 [ %max154, %bb359 ], [ %L492b.0, %bb382 ]		; <i32> [#uses=1]
	%before.pn = phi i32 [ %before, %bb359 ], [ 1, %bb382 ]		; <i32> [#uses=1]
	%L492b.0 = add i32 %before.pn, %max379.pn		; <i32> [#uses=3]
	%tmp386 = load %struct.RETURN** %tmp63, align 4		; <%struct.RETURN*> [#uses=1]
	%tmp387 = getelementptr %struct.RETURN* %tmp386, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp388 = load i32* %tmp387, align 4		; <i32> [#uses=2]
	%tmp392 = load i8** %tmp315, align 4		; <i8*> [#uses=2]
	%R493b.0 = add i32 %indvar, %before		; <i32> [#uses=1]
	%tmp405 = sub i32 %R493b.0, %tmp388		; <i32> [#uses=1]
	%tmp406 = getelementptr i8* %tmp392, i32 %tmp405		; <i8*> [#uses=1]
	%tmp407 = load i8* %tmp406, align 1		; <i8> [#uses=1]
	%tmp408 = sub i32 %L492b.0, %tmp388		; <i32> [#uses=1]
	%tmp409 = getelementptr i8* %tmp392, i32 %tmp408		; <i8*> [#uses=1]
	store i8 %tmp407, i8* %tmp409, align 1
	%tmp414 = icmp eq i32 %L492b.0, %tmp155		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp414, label %bb481, label %bb382

bb420:		; preds = %bb84
	br i1 %tmp431, label %bb481, label %bb436.preheader

bb436.preheader:		; preds = %bb420
	%tmp4468 = load i8** %tmp315, align 4		; <i8*> [#uses=2]
	%tmp4599 = sub i32 %tmp148, %tmp330		; <i32> [#uses=1]
	%tmp46010 = getelementptr i8* %tmp4468, i32 %tmp4599		; <i8*> [#uses=1]
	%tmp46111 = load i8* %tmp46010, align 1		; <i8> [#uses=1]
	%tmp46212 = sub i32 %tmp155, %tmp330		; <i32> [#uses=1]
	%tmp46313 = getelementptr i8* %tmp4468, i32 %tmp46212		; <i8*> [#uses=1]
	store i8 %tmp46111, i8* %tmp46313, align 1
	%exitcond14 = icmp eq i32 %tmp155, %tmp324		; <i1> [#uses=1]
	br i1 %exitcond14, label %bb481, label %bb.nph

bb.nph:		; preds = %bb436.preheader
	%tmp5 = sub i32 %tmp148, %before		; <i32> [#uses=1]
	br label %bb478

bb478:		; preds = %bb478, %bb.nph
	%indvar6422 = phi i32 [ 0, %bb.nph ], [ %indvar.next643, %bb478 ]		; <i32> [#uses=1]
	%indvar.next643 = add i32 %indvar6422, 1		; <i32> [#uses=4]
	%L490b.0 = sub i32 %tmp155, %indvar.next643		; <i32> [#uses=1]
	%R491b.0 = sub i32 %tmp148, %indvar.next643		; <i32> [#uses=1]
	%tmp440 = load %struct.RETURN** %tmp63, align 4		; <%struct.RETURN*> [#uses=1]
	%tmp441 = getelementptr %struct.RETURN* %tmp440, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp442 = load i32* %tmp441, align 4		; <i32> [#uses=2]
	%tmp446 = load i8** %tmp315, align 4		; <i8*> [#uses=2]
	%tmp459 = sub i32 %R491b.0, %tmp442		; <i32> [#uses=1]
	%tmp460 = getelementptr i8* %tmp446, i32 %tmp459		; <i8*> [#uses=1]
	%tmp461 = load i8* %tmp460, align 1		; <i8> [#uses=1]
	%tmp462 = sub i32 %L490b.0, %tmp442		; <i32> [#uses=1]
	%tmp463 = getelementptr i8* %tmp446, i32 %tmp462		; <i8*> [#uses=1]
	store i8 %tmp461, i8* %tmp463, align 1
	%exitcond = icmp eq i32 %indvar.next643, %tmp5		; <i1> [#uses=1]
	br i1 %exitcond, label %bb481, label %bb478

bb481:		; preds = %bb478, %bb436.preheader, %bb420, %bb382, %bb359
	%tmp577 = add i32 %before, -1		; <i32> [#uses=3]
	%tmp578 = add i32 %max154, %tmp577		; <i32> [#uses=2]
	%tmp581 = icmp sge i32 %tmp578, %tmp577		; <i1> [#uses=1]
	%max582 = select i1 %tmp581, i32 %tmp578, i32 %tmp577		; <i32> [#uses=1]
	%tmp584 = sub i32 %max582, %before		; <i32> [#uses=1]
	%tmp585 = add i32 %tmp584, 1		; <i32> [#uses=2]
	%tmp586 = icmp sgt i32 %tmp585, -1		; <i1> [#uses=1]
	%max587 = select i1 %tmp586, i32 %tmp585, i32 0		; <i32> [#uses=1]
	%tmp591 = load %struct.RETURN** %tmp63, align 4		; <%struct.RETURN*> [#uses=1]
	%tmp592 = getelementptr %struct.RETURN* %tmp591, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp593 = load i32* %tmp592, align 4		; <i32> [#uses=1]
	%tmp597 = load i8** %tmp315, align 4		; <i8*> [#uses=1]
	%tmp600621 = trunc i64 %new_item.0.0 to i32		; <i32> [#uses=1]
	%tmp600622 = inttoptr i32 %tmp600621 to i8*		; <i8*> [#uses=1]
	%tmp601 = sub i32 %before, %tmp593		; <i32> [#uses=1]
	%tmp602 = getelementptr i8* %tmp597, i32 %tmp601		; <i8*> [#uses=1]
	tail call void @llvm.memcpy.i32( i8* %tmp602, i8* %tmp600622, i32 %max587, i32 1 )
	%tmp606 = load i32* %tmp71, align 4		; <i32> [#uses=1]
	%tmp613 = add i32 %tmp606, %max154		; <i32> [#uses=1]
	store i32 %tmp613, i32* %tmp71, align 4
	ret void
}
