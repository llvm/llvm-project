; RUN: opt < %s -globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)

; A vtable with ptrauth'd pointers
@vtable = internal unnamed_addr constant { [2 x i8*] } { [2 x i8*] [
  i8* bitcast ({ i8*, i32, i64, i64 }* @vfunc1_live.ptrauth to i8*),
  i8* bitcast ({ i8*, i32, i64, i64 }* @vfunc2_dead.ptrauth to i8*)
]}, align 8, !type !0, !type !1, !vcall_visibility !{i64 2}
!0 = !{i64 0, !"vfunc1.type"}
!1 = !{i64 8, !"vfunc2.type"}

; CHECK:      @vtable = internal unnamed_addr constant { [2 x i8*] } { [2 x i8*] [
; CHECK-SAME:   i8* bitcast ({ i8*, i32, i64, i64 }* @vfunc1_live.ptrauth to i8*),
; CHECK-SAME:   i8* null
; CHECK-SAME: ] }, align 8, !type !0, !type !1, !vcall_visibility !2

; (1) vfunc1_live is referenced from @main, stays alive
@vfunc1_live.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @vfunc1_live to i8*), i32 0, i64 0, i64 42 }, section "llvm.ptrauth"
define internal void @vfunc1_live() {
  ; CHECK: define internal void @vfunc1_live(
  ret void
}

; (2) vfunc2_dead is never referenced, gets removed and vtable slot is null'd
@vfunc2_dead.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @vfunc2_dead to i8*), i32 0, i64 0, i64 42 }, section "llvm.ptrauth"
define internal void @vfunc2_dead() {
  ; CHECK-NOT: define internal void @vfunc2_dead(
  ret void
}

define void @main() {
  %1 = ptrtoint { [2 x i8*] }* @vtable to i64 ; to keep @vtable alive
  %2 = tail call { i8*, i1 } @llvm.type.checked.load(i8* null, i32 0, metadata !"vfunc1.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
