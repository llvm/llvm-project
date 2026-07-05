; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -codegenprepare %s | FileCheck %s

define internal fastcc void @callee(ptr nocapture %p, i32 %a) #0 {
  store volatile i32 %a, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @func_caller(
; CHECK: tail call fastcc void @callee(
; CHECK-NEXT: ret void
; CHECK: ret void
define void @func_caller(ptr nocapture %p, i32 %a, i32 %b) #0 {
entry:
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %bb, label %ret

bb:
  tail call fastcc void @callee(ptr %p, i32 %a)
  br label %ret

ret:
  ret void
}

; CHECK-LABEL: @kernel_caller(
; CHECK: tail call fastcc void @callee(
; CHECK-NEXT: br label %ret

; CHECK: ret void
define amdgpu_kernel void @kernel_caller(ptr nocapture %p, i32 %a, i32 %b) #0 {
entry:
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %bb, label %ret

bb:
  tail call fastcc void @callee(ptr %p, i32 %a)
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind }
