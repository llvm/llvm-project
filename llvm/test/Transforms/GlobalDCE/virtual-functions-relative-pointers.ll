; RUN: opt < %s -passes=globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

; A vtable with "relative pointers", slots don't contain pointers to implementations, but instead have an i32 offset from the vtable itself to the implementation.
@vtable = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1_live to i64), i64 ptrtoint (ptr @vtable to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2_dead to i64), i64 ptrtoint (ptr @vtable to i64)) to i32)
]}, align 8, !type !0, !type !1, !vcall_visibility !{i64 2}
!0 = !{i64 0, !"vfunc1.type"}
!1 = !{i64 4, !"vfunc2.type"}

; CHECK:      @vtable = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1_live to i64), i64 ptrtoint (ptr @vtable to i64)) to i32),
; CHECK-SAME:   i32 0
; CHECK-SAME: ] }, align 8, !type !0, !type !1, !vcall_visibility !2

; Similar to above, but the vtable is more aligned to how C++ relative vtables look.
; That is, the functions may not be dso-local.
@vtable2 = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vfunc3_live_extern to i64), i64 ptrtoint (ptr @vtable2 to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vfunc4_dead_extern to i64), i64 ptrtoint (ptr @vtable2 to i64)) to i32)
]}, align 4, !type !3, !type !4, !vcall_visibility !{i64 2}
!3 = !{i64 0, !"vfunc3.type"}
!4 = !{i64 4, !"vfunc4.type"}

; CHECK:      @vtable2 = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vfunc3_live_extern to i64), i64 ptrtoint (ptr @vtable2 to i64)) to i32),
; CHECK-SAME:   i32 0
; CHECK-SAME: ] }, align 4, !type !3, !type !4, !vcall_visibility !2

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

; (3) vfunc3_live_extern is referenced from @main, stays alive
; CHECK: declare void @vfunc3_live_extern
declare void @vfunc3_live_extern()

; (4) vfunc4_dead_extern is never referenced, gets removed and vtable slot is null'd
; CHECK-NOT: declare void @vfunc4_dead_extern
declare void @vfunc4_dead_extern()

define void @main() {
  %1 = ptrtoint ptr @vtable to i64 ; to keep @vtable alive
  %2 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc1.type")
  %3 = ptrtoint ptr @vtable2 to i64 ; to keep @vtable2 alive
  %4 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc3.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
