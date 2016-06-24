; RUN: opt -S -mtriple=amdgcn-- -amdgpu-codegenprepare < %s | FileCheck %s
; RUN: opt -S -amdgpu-codegenprepare < %s
; Make sure this doesn't crash with no triple

; CHECK-LABEL: @foo(
define void @foo() {
  ret void
}
