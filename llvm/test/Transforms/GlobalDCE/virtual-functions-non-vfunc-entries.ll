; RUN: opt < %s -globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)

; A vtable that contains a non-nfunc entry, @regular_non_virtual_funcA, but
; without a range specific in !vcall_visibility, which means *all* function
; pointers are eligible for VFE, so GlobalDCE will treat the
; @regular_non_virtual_funcA slot as eligible for VFE, and remove it.
@vtableA = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [
  i8* bitcast (void ()* @vfunc1_live to i8*),
  i8* bitcast (void ()* @vfunc2_dead to i8*),
  i8* bitcast (void ()* @regular_non_virtual_funcA to i8*)
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; CHECK:      @vtableA = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [
; CHECK-SAME:   i8* bitcast (void ()* @vfunc1_live to i8*),
; CHECK-SAME:   i8* null,
; CHECK-SAME:   i8* null
; CHECK-SAME: ] }, align 8


; A vtable that contains a non-nfunc entry, @regular_non_virtual_funcB, with a
; range of [0,16) which means only the first two entries are eligible for VFE.
; GlobalDCE should keep @regular_non_virtual_funcB in the vtable.
@vtableB = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [
  i8* bitcast (void ()* @vfunc1_live to i8*),
  i8* bitcast (void ()* @vfunc2_dead to i8*),
  i8* bitcast (void ()* @regular_non_virtual_funcB to i8*)
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2, i64 0, i64 16}

; CHECK:      @vtableB = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [
; CHECK-SAME:   i8* bitcast (void ()* @vfunc1_live to i8*),
; CHECK-SAME:   i8* null,
; CHECK-SAME:   i8* bitcast (void ()* @regular_non_virtual_funcB to i8*)
; CHECK-SAME: ] }, align 8

; A vtable that contains a non-nfunc entry, @regular_non_virtual_funcB, with a
; range of [0,16) which means only the first two entries are eligible for VFE.
; GlobalDCE should keep @regular_non_virtual_funcB in the vtable.
@vtableC = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [
  i8* bitcast (void ()* @regular_non_virtual_funcC to i8*),
  i8* bitcast (void ()* @vfunc1_live to i8*),
  i8* bitcast (void ()* @vfunc2_dead to i8*)
]}, align 8, !type !{i64 8, !"vfunc1.type"}, !type !{i64 16, !"vfunc2.type"}, !vcall_visibility !{i64 2, i64 8, i64 24}

; CHECK:      @vtableC = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [
; CHECK-SAME:   i8* bitcast (void ()* @regular_non_virtual_funcC to i8*),
; CHECK-SAME:   i8* bitcast (void ()* @vfunc1_live to i8*),
; CHECK-SAME:   i8* null
; CHECK-SAME: ] }, align 8

; A vtable with "relative pointers"
@vtableD = internal unnamed_addr constant { i32, i32, i8* } {
  i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc1_live              to i64), i64 ptrtoint ({ i32, i32, i8* }* @vtableD to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc2_dead              to i64), i64 ptrtoint ({ i32, i32, i8* }* @vtableD to i64)) to i32),
  i8* bitcast (void ()* @regular_non_virtual_funcD to i8*)
}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 4, !"vfunc2.type"}, !vcall_visibility !{i64 2, i64 0, i64 8}

; CHECK:      @vtableD = internal unnamed_addr constant { i32, i32, i8* } {
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (void ()* @vfunc1_live              to i64), i64 ptrtoint ({ i32, i32, i8* }* @vtableD to i64)) to i32),
; CHECK-SAME:   i32 0,
; CHECK-SAME:   i8* bitcast (void ()* @regular_non_virtual_funcD to i8*)
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
  %1 = ptrtoint { [3 x i8*] }* @vtableA to i64 ; to keep @vtableA alive
  %2 = ptrtoint { [3 x i8*] }* @vtableB to i64 ; to keep @vtableB alive
  %3 = ptrtoint { [3 x i8*] }* @vtableC to i64 ; to keep @vtableC alive
  %4 = ptrtoint { i32, i32, i8* }* @vtableD to i64 ; to keep @vtableD alive
  %5 = tail call { i8*, i1 } @llvm.type.checked.load(i8* null, i32 0, metadata !"vfunc1.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
