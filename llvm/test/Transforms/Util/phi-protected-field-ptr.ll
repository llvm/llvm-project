; RUN: opt -O2 -S < %s | FileCheck %s

; Test that no optimization run at -O2 moves the loads into the exit block,
; as this causes unnecessary address escapes with pointer field protection.

target triple = "aarch64-unknown-linux-gnu"

define ptr @phi_prot_ptr(i1 %sel, ptr %p1, ptr %p2) {
  br i1 %sel, label %t, label %f

; CHECK: t:
t:
  ; CHECK-NEXT: call
  %protp1 = call ptr @llvm.protected.field.ptr(ptr %p1, i64 1, i1 true)
  ; CHECK-NEXT: load
  %load1 = load ptr, ptr %protp1
  br label %exit

; CHECK: f:
f:
  ; CHECK-NEXT: call
  %protp2 = call ptr @llvm.protected.field.ptr(ptr %p2, i64 2, i1 true)
  ; CHECK-NEXT: load
  %load2 = load ptr, ptr %protp2
  br label %exit

; CHECK: exit:
exit:
  ; CHECK-NEXT: phi
  %retval = phi ptr [ %load1, %t ], [ %load2, %f ]
  ; CHECK-NEXT: ret
  ret ptr %retval
}
