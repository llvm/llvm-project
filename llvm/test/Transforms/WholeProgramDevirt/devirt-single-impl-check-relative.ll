; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: <unknown>:0:0: single-impl: devirtualized a call to vf
; CHECK: remark: <unknown>:0:0: devirtualized vf
; CHECK-NOT: devirtualized

; A vtable with "relative pointers", slots don't contain pointers to implementations, but instead have an i32 offset from the vtable itself to the implementation.
@vtable = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc1_live to i64), i64 ptrtoint (ptr @vtable to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vfunc2_dead to i64), i64 ptrtoint (ptr @vtable to i64)) to i32)
]}, align 8, !type !0, !type !1
;, !vcall_visibility !{i64 2}
!0 = !{i64 0, !"vfunc1.type"}
!1 = !{i64 4, !"vfunc2.type"}

define internal void @vfunc1_live() {
  ret void
}

define internal void @vfunc2_dead() {
  ret void
}

; CHECK: define void @call
define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr %vtable, i32 0, metadata !"vfunc1.type")
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

declare {ptr, i1} @llvm.type.checked.load.relative(ptr, i32, metadata)
declare void @llvm.trap()
