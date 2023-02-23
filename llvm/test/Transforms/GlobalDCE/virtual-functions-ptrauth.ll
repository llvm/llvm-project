; RUN: opt -S -passes=globaldce < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

; A vtable with ptrauth'd pointers
@vtable = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
  ptr @vfunc1_live.ptrauth,
  ptr @vfunc2_dead.ptrauth 
]}, align 8, !type !0, !type !1, !vcall_visibility !{i64 2}
!0 = !{i64 0, !"vfunc1.type"}
!1 = !{i64 8, !"vfunc2.type"}

; CHECK:      @vtable = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
; CHECK-SAME:   ptr @vfunc1_live.ptrauth,
; CHECK-SAME:   ptr null
; CHECK-SAME: ] }, align 8, !type !0, !type !1, !vcall_visibility !2

; (1) vfunc1_live is referenced from @main, stays alive
@vfunc1_live.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @vfunc1_live , i32 0, i64 0, i64 42 }, section "llvm.ptrauth"
define internal void @vfunc1_live() {
  ; CHECK: define internal void @vfunc1_live(
  ret void
}

; (2) vfunc2_dead is never referenced, gets removed and vtable slot is null'd
@vfunc2_dead.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @vfunc2_dead , i32 0, i64 0, i64 42 }, section "llvm.ptrauth"
define internal void @vfunc2_dead() {
  ; CHECK-NOT: define internal void @vfunc2_dead(
  ret void
}

define void @main() {
  %1 = ptrtoint ptr @vtable to i64 ; to keep @vtable alive
  %2 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc1.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
