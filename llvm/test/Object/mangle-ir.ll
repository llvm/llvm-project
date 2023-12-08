; RUN: llvm-as %s -o - | llvm-nm - | FileCheck %s

target datalayout = "m:o"

; CHECK-NOT: memcpy
; CHECK: T _f
; CHECK-NOT: memcpy

define void @f() {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* null, i8* null, i64 0, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)
