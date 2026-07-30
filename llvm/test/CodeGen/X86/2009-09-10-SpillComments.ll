; RUN: llc < %s -mtriple=x86_64-unknown-linux | FileCheck %s

; This test shouldn't require spills.

; CHECK: pushq
; CHECK-NOT: $rsp
; CHECK: popq

	%struct..0anon = type { i32 }
	%struct.rtvec_def = type { i32, [1 x %struct..0anon] }
	%struct.rtx_def = type { i16, i8, i8, [1 x %struct..0anon] }
@rtx_format = external global [116 x ptr]		; <ptr> [#uses=1]
@rtx_length = external global [117 x i32]		; <ptr> [#uses=1]

declare ptr @fixup_memory_subreg(ptr, ptr, i32)

define ptr @walk_fixup_memory_subreg(ptr %x, ptr %insn) {
entry:
	%tmp2 = icmp eq ptr %x, null		; <i1> [#uses=1]
	br i1 %tmp2, label %UnifiedReturnBlock, label %cond_next

cond_next:		; preds = %entry
	%tmp6 = getelementptr %struct.rtx_def, ptr %x, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp7 = load i16, ptr %tmp6		; <i16> [#uses=2]
	%tmp78 = zext i16 %tmp7 to i32		; <i32> [#uses=2]
	%tmp10 = icmp eq i16 %tmp7, 54		; <i1> [#uses=1]
	br i1 %tmp10, label %cond_true13, label %cond_next32

cond_true13:		; preds = %cond_next
	%tmp15 = getelementptr %struct.rtx_def, ptr %x, i32 0, i32 3		; <ptr> [#uses=1]
	%tmp19 = load ptr, ptr %tmp15		; <ptr> [#uses=1]
	%tmp20 = getelementptr %struct.rtx_def, ptr %tmp19, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp21 = load i16, ptr %tmp20		; <i16> [#uses=1]
	%tmp22 = icmp eq i16 %tmp21, 57		; <i1> [#uses=1]
	br i1 %tmp22, label %cond_true25, label %cond_next32

cond_true25:		; preds = %cond_true13
	%tmp29 = tail call ptr @fixup_memory_subreg( ptr %x, ptr %insn, i32 1 ) nounwind		; <ptr> [#uses=1]
	ret ptr %tmp29

cond_next32:		; preds = %cond_true13, %cond_next
	%tmp34 = getelementptr [116 x ptr], ptr @rtx_format, i32 0, i32 %tmp78		; <ptr> [#uses=1]
	%tmp35 = load ptr, ptr %tmp34, align 4		; <ptr> [#uses=1]
	%tmp37 = getelementptr [117 x i32], ptr @rtx_length, i32 0, i32 %tmp78		; <ptr> [#uses=1]
	%tmp38 = load i32, ptr %tmp37, align 4		; <i32> [#uses=1]
	%i.011 = add i32 %tmp38, -1		; <i32> [#uses=2]
	%tmp12513 = icmp sgt i32 %i.011, -1		; <i1> [#uses=1]
	br i1 %tmp12513, label %bb, label %UnifiedReturnBlock

bb:		; preds = %bb123, %cond_next32
	%indvar = phi i32 [ %indvar.next26, %bb123 ], [ 0, %cond_next32 ]		; <i32> [#uses=2]
	%i.01.0 = sub i32 %i.011, %indvar		; <i32> [#uses=5]
	%tmp42 = getelementptr i8, ptr %tmp35, i32 %i.01.0		; <ptr> [#uses=2]
	%tmp43 = load i8, ptr %tmp42		; <i8> [#uses=1]
	switch i8 %tmp43, label %bb123 [
		 i8 101, label %cond_true47
		 i8 69, label %bb105.preheader
	]

cond_true47:		; preds = %bb
	%tmp52 = getelementptr %struct.rtx_def, ptr %x, i32 0, i32 3, i32 %i.01.0		; <ptr> [#uses=1]
	%tmp55 = load ptr, ptr %tmp52		; <ptr> [#uses=1]
	%tmp58 = tail call  ptr @walk_fixup_memory_subreg( ptr %tmp55, ptr %insn ) nounwind		; <ptr> [#uses=1]
	%tmp62 = getelementptr %struct.rtx_def, ptr %x, i32 0, i32 3, i32 %i.01.0, i32 0		; <ptr> [#uses=1]
	%tmp58.c = ptrtoint ptr %tmp58 to i32		; <i32> [#uses=1]
	store i32 %tmp58.c, ptr %tmp62
	%tmp6816 = load i8, ptr %tmp42		; <i8> [#uses=1]
	%tmp6917 = icmp eq i8 %tmp6816, 69		; <i1> [#uses=1]
	br i1 %tmp6917, label %bb105.preheader, label %bb123

bb105.preheader:		; preds = %cond_true47, %bb
	%tmp11020 = getelementptr %struct.rtx_def, ptr %x, i32 0, i32 3, i32 %i.01.0		; <ptr> [#uses=1]
	%tmp11322 = load ptr, ptr %tmp11020		; <ptr> [#uses=1]
	%tmp11423 = getelementptr %struct.rtvec_def, ptr %tmp11322, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp11524 = load i32, ptr %tmp11423		; <i32> [#uses=1]
	%tmp11625 = icmp eq i32 %tmp11524, 0		; <i1> [#uses=1]
	br i1 %tmp11625, label %bb123, label %bb73

bb73:		; preds = %bb73, %bb105.preheader
	%j.019 = phi i32 [ %tmp104, %bb73 ], [ 0, %bb105.preheader ]		; <i32> [#uses=3]
	%tmp81 = load ptr, ptr %tmp11020		; <ptr> [#uses=2]
	%tmp92 = getelementptr %struct.rtvec_def, ptr %tmp81, i32 0, i32 1, i32 %j.019		; <ptr> [#uses=1]
	%tmp95 = load ptr, ptr %tmp92		; <ptr> [#uses=1]
	%tmp98 = tail call  ptr @walk_fixup_memory_subreg( ptr %tmp95, ptr %insn ) nounwind		; <ptr> [#uses=1]
	%tmp101 = getelementptr %struct.rtvec_def, ptr %tmp81, i32 0, i32 1, i32 %j.019, i32 0		; <ptr> [#uses=1]
	%tmp98.c = ptrtoint ptr %tmp98 to i32		; <i32> [#uses=1]
	store i32 %tmp98.c, ptr %tmp101
	%tmp104 = add i32 %j.019, 1		; <i32> [#uses=2]
	%tmp113 = load ptr, ptr %tmp11020		; <ptr> [#uses=1]
	%tmp114 = getelementptr %struct.rtvec_def, ptr %tmp113, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp115 = load i32, ptr %tmp114		; <i32> [#uses=1]
	%tmp116 = icmp ult i32 %tmp104, %tmp115		; <i1> [#uses=1]
	br i1 %tmp116, label %bb73, label %bb123

bb123:		; preds = %bb73, %bb105.preheader, %cond_true47, %bb
	%i.0 = add i32 %i.01.0, -1		; <i32> [#uses=1]
	%tmp125 = icmp sgt i32 %i.0, -1		; <i1> [#uses=1]
	%indvar.next26 = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp125, label %bb, label %UnifiedReturnBlock

UnifiedReturnBlock:		; preds = %bb123, %cond_next32, %entry
	%UnifiedRetVal = phi ptr [ null, %entry ], [ %x, %cond_next32 ], [ %x, %bb123 ]		; <ptr> [#uses=1]
	ret ptr %UnifiedRetVal
}
