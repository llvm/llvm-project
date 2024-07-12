; RUN: llvm-as %s -o - | llvm-nm - | FileCheck %s

target datalayout = "m:o"

; CHECK-NOT: memcpy
; CHECK: T _f
; CHECK-NOT: memcpy

define void @f() {
  tail call void @llvm.memcpy.p0.p0.i64(ptr null, ptr null, i64 0, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)
