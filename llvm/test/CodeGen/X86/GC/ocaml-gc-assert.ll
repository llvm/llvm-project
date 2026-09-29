; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; PR3168

; CHECK-LABEL: append

define ptr @append() gc "ocaml" {
entry:
  switch i32 0, label %L2 [i32 0, label %L1]
L1:
  %var8 = alloca ptr
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
