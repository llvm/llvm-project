; RUN: opt < %s -passes='globaldce' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)

; A vtable with "relative pointers", slots don't contain pointers to implementations, but instead have an i32 offset from the vtable itself to the implementation.
@vtable = internal unnamed_addr constant { [4 x i32] } { [4 x i32] [
  i32 42,
  i32 1337,
  i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc1_live              to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @vtable, i32 0, i32 0, i32 2) to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc2_dead              to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @vtable, i32 0, i32 0, i32 2) to i64)) to i32)
]}, align 8, !type !0, !type !1, !vcall_visibility !{i64 2}
!0 = !{i64 8, !"vfunc1.type"}
!1 = !{i64 12, !"vfunc2.type"}

; CHECK:      @vtable = internal unnamed_addr constant { [4 x i32] } { [4 x i32] [
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc1_live              to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @vtable, i32 0, i32 0, i32 2) to i64)) to i32),
; CHECK-SAME:   i32 0
; CHECK-SAME: ] }, align 8, !type !0, !type !1, !vcall_visibility !2

; (1) vfunc1_live is referenced from @main, stays alive
define internal void @vfunc1_live() {
  ; CHECK: define internal void @vfunc1_live(
  ret void
}

; (2) vfunc2_dead is never referenced, gets removed and vtable slot is null'd
define internal void @vfunc2_dead() {
  ; CHECK-NOT: define internal void @vfunc2_dead(
  ret void
}

define void @main() {
  %1 = ptrtoint { [4 x i32] }* @vtable to i64 ; to keep @vtable alive
  %2 = tail call { i8*, i1 } @llvm.type.checked.load(i8* null, i32 0, metadata !"vfunc1.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
