; REQUIRES: x86-registered-target
; RUN: opt -S -passes=kcfi %s | FileCheck %s

;; If the back-end supports KCFI operand bundle lowering, KCFIPass should be a no-op.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define void @f1
define void @f1(ptr noundef %x) {
  ; CHECK-NOT: call void @llvm.trap()
  ; CHECK: call void %x() [ "kcfi"(i32 12345678) ]
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
