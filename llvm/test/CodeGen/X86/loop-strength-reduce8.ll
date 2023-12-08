; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s

; FIXME: The first two instructions, movl and addl, should have been combined to
; "leal 16(%eax), %edx" by the backend (PR20776).
; CHECK: movl    %eax, %edx
; CHECK: addl    $16, %edx
; CHECK: align
; CHECK: addl    $4, %edx
; CHECK: decl    %ecx
; CHECK: jne     LBB0_2

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32 }
	%struct.bitmap_element = type { ptr, ptr, i32, [2 x i64] }
	%struct.bitmap_head_def = type { ptr, ptr, i32 }
	%struct.branch_path = type { ptr, i32 }
	%struct.c_lang_decl = type <{ i8, [3 x i8] }>
	%struct.constant_descriptor = type { ptr, ptr, ptr, { x86_fp80 } }
	%struct.eh_region = type { ptr, ptr, ptr, i32, ptr, i32, { { ptr, ptr, ptr, ptr } }, ptr, ptr, ptr, ptr }
	%struct.eh_status = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }
	%struct.emit_status = type { i32, i32, ptr, ptr, ptr, ptr, i32, i32, ptr, i32, ptr, ptr, ptr }
	%struct.equiv_table = type { ptr, ptr }
	%struct.expr_status = type { i32, i32, i32, ptr, ptr, ptr, ptr }
	%struct.function = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, %struct.CUMULATIVE_ARGS, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, ptr, ptr, ptr, i32, i32, ptr, i32, i32, ptr, ptr, i8, i8, i8 }
	%struct.goto_fixup = type { ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr }
	%struct.initial_value_struct = type { i32, i32, ptr }
	%struct.label_chain = type { ptr, ptr }
	%struct.lang_decl = type { %struct.c_lang_decl, ptr }
	%struct.language_function = type { %struct.stmt_tree_s, ptr }
	%struct.machine_function = type { [59 x [3 x ptr]], i32, i32 }
	%struct.nesting = type { ptr, ptr, i32, ptr, { { i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, ptr } } }
	%struct.pool_constant = type { ptr, ptr, ptr, ptr, i32, i32, i32, i64, i32 }
	%struct.rtunion = type { i64 }
	%struct.rtx_def = type { i16, i8, i8, [1 x %struct.rtunion] }
	%struct.sequence_stack = type { ptr, ptr, ptr, ptr }
	%struct.stmt_status = type { ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, i32, ptr, i32, ptr }
	%struct.stmt_tree_s = type { ptr, ptr, ptr, i32 }
	%struct.temp_slot = type { ptr, ptr, ptr, i32, i64, ptr, ptr, i8, i8, i32, i32, i64, i64 }
	%struct.tree_common = type { ptr, ptr, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, ptr, i32, i32, ptr, i8, i8, i8, i8, i8, i8, %struct.rtunion, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, { ptr }, ptr, ptr, ptr, i64, ptr }
	%struct.tree_exp = type { %struct.tree_common, i32, [1 x ptr] }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.var_refs_queue = type { ptr, i32, i32, ptr }
	%struct.varasm_status = type { ptr, ptr, ptr, ptr, i64, ptr }
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i32, i32, i32, ptr, %struct.varray_data }
@lineno = internal global i32 0		; <ptr> [#uses=1]
@tree_code_length = internal global [256 x i32] zeroinitializer
@llvm.used = appending global [1 x ptr] [ ptr @build_stmt ], section "llvm.metadata"		; <ptr> [#uses=0]

define ptr @build_stmt(i32 %code, ...) nounwind {
entry:
	%p = alloca ptr		; <ptr> [#uses=3]
	call void @llvm.va_start(ptr %p)
	%0 = call fastcc ptr @make_node(i32 %code) nounwind		; <ptr> [#uses=2]
	%1 = getelementptr [256 x i32], ptr @tree_code_length, i32 0, i32 %code		; <ptr> [#uses=1]
	%2 = load i32, ptr %1, align 4		; <i32> [#uses=2]
	%3 = load i32, ptr @lineno, align 4		; <i32> [#uses=1]
	%4 = getelementptr %struct.tree_exp, ptr %0, i32 0, i32 1		; <ptr> [#uses=1]
	store i32 %3, ptr %4, align 4
	%5 = icmp sgt i32 %2, 0		; <i1> [#uses=1]
	br i1 %5, label %bb, label %bb3

bb:		; preds = %bb, %entry
	%i.01 = phi i32 [ %indvar.next, %bb ], [ 0, %entry ]		; <i32> [#uses=2]
	%6 = load ptr, ptr %p, align 4		; <ptr> [#uses=2]
	%7 = getelementptr i8, ptr %6, i32 4		; <ptr> [#uses=1]
	store ptr %7, ptr %p, align 4
	%8 = load ptr, ptr %6, align 4		; <ptr> [#uses=1]
	%9 = getelementptr %struct.tree_exp, ptr %0, i32 0, i32 2, i32 %i.01		; <ptr> [#uses=1]
	store ptr %8, ptr %9, align 4
	%indvar.next = add i32 %i.01, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %2		; <i1> [#uses=1]
	br i1 %exitcond, label %bb3, label %bb

bb3:		; preds = %bb, %entry
	call void @llvm.va_end(ptr %p)
	ret ptr %0
}

declare void @llvm.va_start(ptr) nounwind

declare void @llvm.va_end(ptr) nounwind

declare fastcc ptr @make_node(i32) nounwind
