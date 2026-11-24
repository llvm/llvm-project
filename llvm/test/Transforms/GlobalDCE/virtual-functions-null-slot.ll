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


; The slot is a byte array that is not all zeroes, this vtable is disqualified from VFE.
; CHECK:      @vtableNonZeroBytes = internal unnamed_addr constant <{ [8 x i8], ptr }> <{
; CHECK-SAME:   [8 x i8] c"\01\00\00\00\00\00\00\00",
; CHECK-SAME:   ptr @vfunc2NonZeroBytes
; CHECK-SAME: }>, align 8
@vtableNonZeroBytes = internal unnamed_addr constant <{ [8 x i8], ptr }> <{
  [8 x i8] c"\01\00\00\00\00\00\00\00",
  ptr @vfunc2NonZeroBytes
}>, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; The slot is a pointer, but not null and not a function, this vtable is disqualified from VFE.
; CHECK:      @vtableIntToPtr = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
; CHECK-SAME:   ptr inttoptr (i64 1 to ptr),
; CHECK-SAME:   ptr @vfunc2IntToPtr
; CHECK-SAME: ] }, align 8
@vtableIntToPtr = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
  ptr inttoptr (i64 1 to ptr),
  ptr @vfunc2IntToPtr
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; Load out of bounds, this vtable is disqualified from VFE.
; CHECK:      @vtableOutOfBounds = internal unnamed_addr constant <{ ptr, i32 }> <{
; CHECK-SAME:   ptr @vfunc2OutOfBounds,
; CHECK-SAME:   i32 0
; CHECK-SAME: }>, align 8
@vtableOutOfBounds = internal unnamed_addr constant <{ ptr, i32 }> <{
  ptr @vfunc2OutOfBounds,
  i32 0
}>, align 8, !type !{i64 0, !"vfunc4.type"}, !vcall_visibility !{i64 2}

; Non-constant vtable, this vtable is disqualified from VFE.
; CHECK:      @vtableMutable = internal unnamed_addr global <{ [8 x i8], ptr }> <{
; CHECK-SAME:   [8 x i8] zeroinitializer,
; CHECK-SAME:   ptr @vfunc2Mutable
; CHECK-SAME: }>, align 8
@vtableMutable = internal unnamed_addr global <{ [8 x i8], ptr }> <{
  [8 x i8] zeroinitializer,
  ptr @vfunc2Mutable
}>, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}


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

; CHECK: define internal void @vfunc2NonZeroBytes(
define internal void @vfunc2NonZeroBytes() {
  ret void
}

; CHECK: define internal void @vfunc2IntToPtr(
define internal void @vfunc2IntToPtr() {
  ret void
}

; CHECK: define internal void @vfunc2OutOfBounds(
define internal void @vfunc2OutOfBounds() {
  ret void
}

; CHECK: define internal void @vfunc2Mutable(
define internal void @vfunc2Mutable() {
  ret void
}

define void @main() {
  ; To keep the vtables alive.
  %1 = ptrtoint ptr @vtableA to i64
  %2 = ptrtoint ptr @vtableB to i64
  %3 = ptrtoint ptr @vtableC to i64
  %4 = ptrtoint ptr @vtableRust to i64
  %5 = ptrtoint ptr @vtableD to i64
  %6 = ptrtoint ptr @vtableE to i64
  %7 = ptrtoint ptr @vtableNonZeroBytes to i64
  %8 = ptrtoint ptr @vtableIntToPtr to i64
  %9 = ptrtoint ptr @vtableOutOfBounds to i64
  %10 = ptrtoint ptr @vtableMutable to i64

  ; Methods targeted by a checked load must be kept.
  %11 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc1.type")
  %12 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 0, metadata !"vfunc3.type")
  %13 = tail call { ptr, i1 } @llvm.type.checked.load(ptr null, i32 8, metadata !"vfunc4.type")

  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
