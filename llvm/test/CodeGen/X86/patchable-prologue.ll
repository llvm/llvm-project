; RUN: llc -verify-machineinstrs -filetype=obj -o - -mtriple=x86_64-apple-macosx < %s | llvm-objdump -triple x86_64-apple-macosx -disassemble - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-apple-macosx < %s | FileCheck %s --check-prefix=CHECK-ALIGN

declare void @callee(i64*)

define void @f0() "patchable-function"="prologue-short-redirect" {
; CHECK-LABEL: _f0:
; CHECK-NEXT:  66 90 	nop

; CHECK-ALIGN: 	.p2align	4, 0x90
; CHECK-ALIGN: _f0:

  ret void
}

define void @f1() "patchable-function"="prologue-short-redirect" "frame-pointer"="all" {
; CHECK-LABEL: _f1
; CHECK-NEXT: ff f5 	pushq	%rbp

; CHECK-ALIGN: 	.p2align	4, 0x90
; CHECK-ALIGN: _f1:
  ret void
}

define void @f2() "patchable-function"="prologue-short-redirect" {
; CHECK-LABEL: _f2
; CHECK-NEXT: 48 81 ec a8 00 00 00 	subq	$168, %rsp

; CHECK-ALIGN: 	.p2align	4, 0x90
; CHECK-ALIGN: _f2:
  %ptr = alloca i64, i32 20
  call void @callee(i64* %ptr)
  ret void
}

define void @f3() "patchable-function"="prologue-short-redirect" optsize {
; CHECK-LABEL: _f3
; CHECK-NEXT: 66 90 	nop

; CHECK-ALIGN: 	.p2align	4, 0x90
; CHECK-ALIGN: _f3:
  ret void
}

; This testcase happens to produce a KILL instruction at the beginning of the
; first basic block. In this case the 2nd instruction should be turned into a
; patchable one.
; CHECK-LABEL: f4:
; CHECK-NEXT: 8b 0c 37  movl  (%rdi,%rsi), %ecx
define i32 @f4(i8* %arg1, i64 %arg2, i32 %arg3) "patchable-function"="prologue-short-redirect" {
bb:
  %tmp10 = getelementptr i8, i8* %arg1, i64 %arg2
  %tmp11 = bitcast i8* %tmp10 to i32*
  %tmp12 = load i32, i32* %tmp11, align 4
  fence acquire
  %tmp13 = add i32 %tmp12, %arg3
  %tmp14 = cmpxchg i32* %tmp11, i32 %tmp12, i32 %tmp13 seq_cst monotonic
  %tmp15 = extractvalue { i32, i1 } %tmp14, 1
  br i1 %tmp15, label %bb21, label %bb16

bb16:
  br label %bb21

bb21:
  %tmp22 = phi i32 [ %tmp12, %bb ], [ %arg3, %bb16 ]
  ret i32 %tmp22
}
