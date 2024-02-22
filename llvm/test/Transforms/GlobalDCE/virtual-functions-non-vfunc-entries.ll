; RUN: opt -S -passes=globaldce < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

; A vtable that contains a non-nfunc entry, @regular_non_virtual_funcA, but
; without a range specific in !vcall_visibility, which means *all* function
; pointers are eligible for VFE, so GlobalDCE will treat the
; @regular_non_virtual_funcA slot as eligible for VFE, and remove it.
@vtableA = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [
  ptr @vfunc1_live,
  ptr @vfunc2_dead,
  ptr @regular_non_virtual_funcA 
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; CHECK:      @vtableA = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [
; CHECK-SAME:   ptr @vfunc1_live,
; CHECK-SAME:   ptr null,
; CHECK-SAME:   ptr null
; CHECK-SAME: ] }, align 8


; A vtable that contains a non-nfunc entry, @regular_non_virtual_funcB, with a
; range of [0,16) which means only the first two entries are eligible for VFE.
; GlobalDCE should keep @regular_non_virtual_funcB in the vtable.
@vtableB = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [
  ptr @vfunc1_live,
  ptr @vfunc2_dead,
  ptr @regular_non_virtual_funcB 
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2, i64 0, i64 16}

; CHECK:      @vtableB = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [
; CHECK-SAME:   ptr @vfunc1_live,
; CHECK-SAME:   ptr null,
; CHECK-SAME:   ptr @regular_non_virtual_funcB 
; CHECK-SAME: ] }, align 8

; A vtable that contains a non-nfunc entry, @regular_non_virtual_funcB, with a
; range of [0,16) which means only the first two entries are eligible for VFE.
; GlobalDCE should keep @regular_non_virtual_funcB in the vtable.
@vtableC = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [
  ptr @regular_non_virtual_funcC,
  ptr @vfunc1_live,
  ptr @vfunc2_dead 
]}, align 8, !type !{i64 8, !"vfunc1.type"}, !type !{i64 16, !"vfunc2.type"}, !vcall_visibility !{i64 2, i64 8, i64 24}

; CHECK:      @vtableC = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [
; CHECK-SAME:   ptr @regular_non_virtual_funcC,
; CHECK-SAME:   ptr @vfunc1_live,
; CHECK-SAME:   ptr null
; CHECK-SAME: ] }, align 8

; A vtable with "relative pointers"
@vtableD = internal unnamed_addr constant { i32, i32, ptr } {
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1_live              to i64), i64 ptrtoint (ptr @vtableD to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2_dead              to i64), i64 ptrtoint (ptr @vtableD to i64)) to i32),
  ptr @regular_non_virtual_funcD 
}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 4, !"vfunc2.type"}, !vcall_visibility !{i64 2, i64 0, i64 8}

; CHECK:      @vtableD = internal unnamed_addr constant { i32, i32, ptr } {
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1_live              to i64), i64 ptrtoint (ptr @vtableD to i64)) to i32),
; CHECK-SAME:   i32 0,
; CHECK-SAME:   ptr @regular_non_virtual_funcD 
; CHECK-SAME: }, align 8

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

; (3) not using a range in !vcall_visibility, global gets removed
define internal void @regular_non_virtual_funcA() {
  ; CHECK-NOT: define internal void @regular_non_virtual_funcA(
  ret void
}

; (4) using a range in !vcall_visibility, pointer is outside of range, so should
; stay alive
define internal void @regular_non_virtual_funcB() {
  ; CHECK: define internal void @regular_non_virtual_funcB(
  ret void
}

; (5) using a range in !vcall_visibility, pointer is outside of range, so should
; stay alive
define internal void @regular_non_virtual_funcC() {
  ; CHECK: define internal void @regular_non_virtual_funcC(
  ret void
}

; (6) using a range in !vcall_visibility, pointer is outside of range, so should
; stay alive
define internal void @regular_non_virtual_funcD() {
  ; CHECK: define internal void @regular_non_virtual_funcD(
  ret void
}

define void @main() {
  %1 = ptrtoint { [3 x ptr] }* @vtableA to i64 ; to keep @vtableA alive
  %2 = ptrtoint { [3 x ptr] }* @vtableB to i64 ; to keep @vtableB alive
  %3 = ptrtoint { [3 x ptr] }* @vtableC to i64 ; to keep @vtableC alive
  %4 = ptrtoint { i32, i32, ptr }* @vtableD to i64 ; to keep @vtableD alive
  %5 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc1.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
