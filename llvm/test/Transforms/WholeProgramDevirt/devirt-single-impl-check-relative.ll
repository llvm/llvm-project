; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: <unknown>:0:0: single-impl: devirtualized a call to vfunc1_live
; CHECK: remark: <unknown>:0:0: single-impl: devirtualized a call to vfunc3_live
; CHECK: remark: <unknown>:0:0: devirtualized vfunc1_live
; CHECK: remark: <unknown>:0:0: devirtualized vfunc3_live
; CHECK-NOT: devirtualized

; A vtable with "relative pointers", slots don't contain pointers to implementations, but instead have an i32 offset from the vtable itself to the implementation.
@vtable = internal unnamed_addr constant { [3 x i32] } { [3 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1_live to i64), i64 ptrtoint (ptr @vtable to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2_dead to i64), i64 ptrtoint (ptr @vtable to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc3_live to i64), i64 ptrtoint (ptr @vtable to i64)) to i32)
]}, align 8, !type !0
!0 = !{i64 0, !"vtable"}

define internal void @vfunc1_live() {
  ret void
}

define internal void @vfunc2_dead() {
  ret void
}

define internal void @vfunc3_live() {
  ret void
}

; CHECK: define void @call_vf1
define void @call_vf1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr %vtable, i32 0, metadata !"vtable")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  ; CHECK: br i1 true,
  br i1 %p, label %cont, label %trap

cont:
  ; CHECK: call void @vfunc1_live(
  call void %fptr()
  ret void

trap:
  call void @llvm.trap()
  unreachable
}

; CHECK: define void @call_vf3
define void @call_vf3(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr %vtable, i32 8, metadata !"vtable")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  ; CHECK: br i1 true,
  br i1 %p, label %cont, label %trap

cont:
  ; CHECK: call void @vfunc3_live(
  call void %fptr()
  ret void

trap:
  call void @llvm.trap()
  unreachable
}

declare {ptr, i1} @llvm.type.checked.load.relative(ptr, i32, metadata)
declare void @llvm.trap()
