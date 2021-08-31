; RUN: opt < %s -globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)

@vtable = internal unnamed_addr constant { [3 x i32] } { [3 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc1              to i64), i64 ptrtoint ({ [3 x i32] }* @vtable to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc2              to i64), i64 ptrtoint ({ [3 x i32] }* @vtable to i64)) to i32),

  ; a "bad" relative pointer because it's base is not the @vtable symbol
  i32 trunc (i64 sub (i64 ptrtoint (void ()* @weird_ref_1         to i64), i64 ptrtoint (void ()* @weird_ref_2 to i64)) to i32)
]}, align 8, !type !0, !type !1, !vcall_visibility !{i64 2}
!0 = !{i64 0, !"vfunc1.type"}
!1 = !{i64 4, !"vfunc2.type"}

; CHECK:      @vtable = internal unnamed_addr constant { [3 x i32] } { [3 x i32] [
; CHECK-SAME:   i32 trunc (i64 sub (i64 0, i64 ptrtoint ({ [3 x i32] }* @vtable to i64)) to i32),
; CHECK-SAME:   i32 trunc (i64 sub (i64 0, i64 ptrtoint ({ [3 x i32] }* @vtable to i64)) to i32),
; CHECK-SAME:   i32 trunc (i64 sub (i64 0, i64 ptrtoint (void ()* @weird_ref_2 to i64)) to i32)
; CHECK-SAME: ] }, align 8, !type !0, !type !1, !vcall_visibility !2

define internal void @vfunc1() { ret void }
define internal void @vfunc2() { ret void }
define internal void @weird_ref_1() { ret void }
define internal void @weird_ref_2() { ret void }

define void @main() {
  %1 = ptrtoint { [3 x i32] }* @vtable to i64 ; to keep @vtable alive
  call void @weird_ref_2()
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
