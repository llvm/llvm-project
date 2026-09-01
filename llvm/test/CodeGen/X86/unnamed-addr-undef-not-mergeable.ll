; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
;
; Regression test for https://github.com/llvm/llvm-project/issues/220091
;
; unnamed_addr constants that contain undef or poison sub-elements must NOT
; be placed in ELF mergeable sections (.rodata.cstN, SHF_MERGE "aM" flag).
;
; The ELF linker merges constants in those sections by comparing raw byte
; content. LLVM emits undef and poison bytes as zeros, which makes such a
; constant byte-identical to a zeroinitializer constant of the same size.
; The linker would then assign them the same address even though they are
; semantically distinct under the LLVM IR spec, leading to miscompilation.
;
; Fix: getKindForGlobal() now returns ReadOnly (plain .rodata) instead of a
; MergeableConst* section kind when the initializer contains undef or poison.

; ---------------------------------------------------------------------------
; @undef_and_zero: struct with undef first half, zero second half (8 bytes).
; Even though it's 8 bytes and unnamed_addr, it must NOT go to .rodata.cst8
; because the undef half would be emitted as zeros, making it byte-identical
; to @all_zero below.
; ---------------------------------------------------------------------------
@undef_and_zero = private unnamed_addr constant <{ [4 x i8], [4 x i8] }> <{ [4 x i8] undef, [4 x i8] zeroinitializer }>, align 4

; CHECK:     .type	.Lundef_and_zero,@object
; CHECK-NOT: .section	.rodata.cst8
; CHECK:     .section	.rodata,"a",@progbits

; ---------------------------------------------------------------------------
; @all_zero: pure zeroinitializer of the same 8-byte size.
; This SHOULD still go to .rodata.cst8 (no regression for non-undef case).
; ---------------------------------------------------------------------------
@all_zero = private unnamed_addr constant i64 0, align 8

; CHECK:     .type	.Lall_zero,@object
; CHECK:     .section	.rodata.cst8,"aM",@progbits,8

; ---------------------------------------------------------------------------
; @nested_undef: undef nested inside a struct field.
; Must be non-mergeable even though it's an aggregate with a concrete field.
; ---------------------------------------------------------------------------
@nested_undef = private unnamed_addr constant { i32, i32 } { i32 undef, i32 42 }, align 4

; CHECK:     .type	.Lnested_undef,@object
; CHECK:     .section	.rodata,"a",@progbits

; ---------------------------------------------------------------------------
; @poison_field: poison value (PoisonValue is a subclass of UndefValue).
; Must also be non-mergeable.
; ---------------------------------------------------------------------------
@poison_field = private unnamed_addr constant { i32, i32 } { i32 poison, i32 0 }, align 4

; CHECK:     .type	.Lpoison_field,@object
; (stays in .rodata from the previous section, no new .section directive needed)

; ---------------------------------------------------------------------------
; @no_undef_4: fully concrete constant -- still gets mergeable section.
; ---------------------------------------------------------------------------
@no_undef_4 = private unnamed_addr constant i32 1234, align 4

; CHECK:     .type	.Lno_undef_4,@object
; CHECK:     .section	.rodata.cst4,"aM",@progbits,4

; Dummy function to prevent globals from being dead-stripped.
define ptr @use() {
  %a = select i1 true,  ptr @undef_and_zero, ptr @all_zero
  %b = select i1 false, ptr @nested_undef,   ptr @poison_field
  %c = select i1 true,  ptr %a,              ptr @no_undef_4
  %d = select i1 false, ptr %b,              ptr %c
  ret ptr %d
}
