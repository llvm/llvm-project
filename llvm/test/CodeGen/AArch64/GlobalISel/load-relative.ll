; RUN: opt < %s -passes=instsimplify -S | llc -mtriple=aarch64 -global-isel -verify-machineinstrs -o - | FileCheck %s

declare void @target()

@vt = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @target to i64), i64 ptrtoint (ptr @vt to i64)) to i32)]

declare ptr @llvm.load.relative.i32(ptr, i32)

; CHECK-LABEL: test:
; CHECK: bl target
define void @test() {
  %fn = call ptr @llvm.load.relative.i32(ptr @vt, i32 0)
  call void %fn()
  ret void
}
