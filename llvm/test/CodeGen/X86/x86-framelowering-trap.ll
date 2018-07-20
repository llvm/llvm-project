; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: bar:
; CHECK: pushq
; CHECK: ud2
; CHECK-NEXT: popq
; CHECK-NEXT: .cfi_def_cfa_offset 8
; CHECK-NEXT: retq
define void @bar() {
entry:
  call void @callee()
  call void @llvm.trap()
  ret void
}

; Function Attrs: noreturn nounwind
declare void @llvm.trap()

declare void @callee()
