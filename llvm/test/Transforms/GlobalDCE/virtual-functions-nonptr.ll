; RUN: opt < %s -globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

@vtableA = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
  i32 0,
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2 to i64), i64 ptrtoint (ptr @vtableA to i64)) to i32)
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 4, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; CHECK:      @vtableA = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
; CHECK-SAME:   i32 0,
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2 to i64), i64 ptrtoint (ptr @vtableA to i64)) to i32)
; CHECK-SAME: ] }, align 8

@vtableB = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1 to i64), i64 ptrtoint (ptr @vtableB to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2 to i64), i64 ptrtoint (ptr @vtableB to i64)) to i32)
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 4, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; CHECK:      @vtableB = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1 to i64), i64 ptrtoint (ptr @vtableB to i64)) to i32),
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2 to i64), i64 ptrtoint (ptr @vtableB to i64)) to i32)
; CHECK-SAME: ] }, align 8

define internal void @vfunc1() {
  ret void
}

define internal void @vfunc2() {
  ret void
}

define void @main() {
  %1 = ptrtoint ptr @vtableA to i64 ; to keep @vtableA alive
  %2 = ptrtoint ptr @vtableB to i64 ; to keep @vtableB alive
  %3 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc1.type")
  %4 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc2.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
