; RUN: llc -filetype=obj -o - -mtriple=x86_64-apple-macosx < %s | llvm-objdump -triple x86_64-apple-macosx -disassemble - | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-macosx < %s | FileCheck %s --check-prefix=CHECK-ALIGN

declare void @callee(i64*)

define void @f0() "patchable-function"="prologue-short-redirect" {
; CHECK-LABEL: _f0:
; CHECK-NEXT:  66 90 	nop

; CHECK-ALIGN: 	.p2align	4, 0x90
; CHECK-ALIGN: _f0:

  ret void
}

define void @f1() "patchable-function"="prologue-short-redirect" "no-frame-pointer-elim"="true" {
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
