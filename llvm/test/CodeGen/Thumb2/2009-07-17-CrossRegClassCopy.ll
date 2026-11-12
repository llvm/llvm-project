; RUN: llc < %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv6t2-elf"
	%struct.dwarf_cie = type <{ i32, i32, i8, [0 x i8], [3 x i8] }>

declare ptr @read_sleb128(ptr, ptr nocapture) nounwind

define i32 @get_cie_encoding(ptr %cie) nounwind {
entry:
	br i1 undef, label %bb1, label %bb13

bb1:		; preds = %entry
	%tmp38 = add i32 undef, 10		; <i32> [#uses=1]
	br label %bb.i

bb.i:		; preds = %bb.i, %bb1
	%indvar.i = phi i32 [ 0, %bb1 ], [ %2, %bb.i ]		; <i32> [#uses=3]
	%tmp39 = add i32 %indvar.i, %tmp38		; <i32> [#uses=1]
	%p_addr.0.i = getelementptr i8, ptr undef, i32 %tmp39		; <ptr> [#uses=1]
	%0 = load i8, ptr %p_addr.0.i, align 1		; <i8> [#uses=1]
	%1 = icmp slt i8 %0, 0		; <i1> [#uses=1]
	%2 = add i32 %indvar.i, 1		; <i32> [#uses=1]
	br i1 %1, label %bb.i, label %read_uleb128.exit

read_uleb128.exit:		; preds = %bb.i
	%.sum40 = add i32 %indvar.i, undef		; <i32> [#uses=1]
	%.sum31 = add i32 %.sum40, 2		; <i32> [#uses=1]
	%scevgep.i = getelementptr %struct.dwarf_cie, ptr %cie, i32 0, i32 3, i32 %.sum31		; <ptr> [#uses=1]
	%3 = call  ptr @read_sleb128(ptr %scevgep.i, ptr undef)		; <ptr> [#uses=0]
	unreachable

bb13:		; preds = %entry
	ret i32 0
}
