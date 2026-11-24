; RUN: opt < %s -passes=globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

; Previously, a NULL slot disqualified a vtable from participating in VFE.
;
; But actually, NULL is not a valid function pointer, so a NULL slot really means that
; the vtable is never used to make a virtual call to the function stored in the slot,
; which is extremely valuable information for VFE.

; CHECK:      @vtableA = internal unnamed_addr constant <{ [8 x i8], ptr }> <{
; CHECK-SAME:   [8 x i8] zeroinitializer,
; CHECK-SAME:   ptr null
; CHECK-SAME: }>, align 8
@vtableA = internal unnamed_addr constant <{ [8 x i8], ptr }> <{
  [8 x i8] zeroinitializer,
  ptr @vfunc2A
}>, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}


; CHECK:      @vtableB = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
; CHECK-SAME:   ptr @vfunc1B,
; CHECK-SAME:   ptr null
; CHECK-SAME: ] }, align 8
@vtableB = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
  ptr @vfunc1B,
  ptr @vfunc2B
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}


; CHECK:      @vtableC = internal unnamed_addr constant { [2 x ptr] } zeroinitializer
@vtableC = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
  ptr null,
  ptr @vfunc2C
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}


; CHECK:      @vtableRust = internal unnamed_addr constant <{ [24 x i8], ptr }> <{
; CHECK-SAME:   [24 x i8] c"\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00",
; CHECK-SAME:   ptr null
; CHECK-SAME: }>, align 8
@vtableRust = internal unnamed_addr constant <{ [24 x i8], ptr }> <{
  [24 x i8] c"\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00",
  ptr @vfunc2Rust
}>, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 24, !"vfunc2.type"}, !vcall_visibility !{i64 2}


; The null slot is at offset 8, the bytes at offset 0 are not null. The !type offset is
; non-zero for @vtableD, and the type.checked.load offset is non-zero for @vtableE.

; CHECK:      @vtableD = internal unnamed_addr constant <{ [16 x i8], ptr }> <{
; CHECK-SAME:   [16 x i8] c"\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00",
; CHECK-SAME:   ptr null
; CHECK-SAME: }>, align 8
@vtableD = internal unnamed_addr constant <{ [16 x i8], ptr }> <{
  [16 x i8] c"\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00",
  ptr @vfunc2D
}>, align 8, !type !{i64 8, !"vfunc3.type"}, !type !{i64 16, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; CHECK:      @vtableE = internal unnamed_addr constant <{ [16 x i8], ptr }> <{
; CHECK-SAME:   [16 x i8] c"\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00",
; CHECK-SAME:   ptr null
; CHECK-SAME: }>, align 8
@vtableE = internal unnamed_addr constant <{ [16 x i8], ptr }> <{
  [16 x i8] c"\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00",
  ptr @vfunc2E
}>, align 8, !type !{i64 0, !"vfunc4.type"}, !type !{i64 16, !"vfunc2.type"}, !vcall_visibility !{i64 2}


; There is a checked load for vfunc1.type, so this function must be kept.
; CHECK: define internal void @vfunc1B(
define internal void @vfunc1B() {
  ret void
}

; There is no checked load for vfunc2.type, so all implementation can be dropped.
; CHECK-NOT: define internal void @vfunc2A(
define internal void @vfunc2A() {
  ret void
}

; CHECK-NOT: define internal void @vfunc2B(
define internal void @vfunc2B() {
  ret void
}

; CHECK-NOT: define internal void @vfunc2C(
define internal void @vfunc2C() {
  ret void
}

; CHECK-NOT: define internal void @vfunc2Rust(
define internal void @vfunc2Rust() {
  ret void
}

; CHECK-NOT: define internal void @vfunc2D(
define internal void @vfunc2D() {
  ret void
}

; CHECK-NOT: define internal void @vfunc2E(
define internal void @vfunc2E() {
  ret void
}

define void @main() {
  %1 = ptrtoint ptr @vtableA to i64 ; to keep @vtableA alive
  %2 = ptrtoint ptr @vtableB to i64 ; to keep @vtableB alive
  %3 = ptrtoint ptr @vtableC to i64 ; to keep @vtableC alive
  %4 = ptrtoint ptr @vtableRust to i64 ; to keep @vtableRust alive
  %5 = ptrtoint ptr @vtableD to i64 ; to keep @vtableD alive
  %6 = ptrtoint ptr @vtableE to i64 ; to keep @vtableE alive
  %7 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc1.type")
  %8 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc3.type")
  %9 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 8, metadata !"vfunc4.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
