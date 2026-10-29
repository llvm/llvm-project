; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; PR3168

; CHECK-LABEL: append

; The root must be a static alloca; the llvm.gcroot call stays in L1, where the
; original test had it.  The switch condition must not be a constant, or the
; CFG folds away before ISel and this stops testing anything.
define ptr @append(i32 %n) gc "ocaml" {
entry:
  %var8 = alloca ptr
  switch i32 %n, label %L2 [i32 0, label %L1]
L1:
  call void @llvm.gcroot(ptr %var8,ptr null)
  br label %L3
L2:
  call ccc void @oread_runtime_casenotcovered()
  unreachable
L3:
  ret ptr null
}

declare ccc void @oread_runtime_casenotcovered()
declare void @llvm.gcroot(ptr,ptr)
