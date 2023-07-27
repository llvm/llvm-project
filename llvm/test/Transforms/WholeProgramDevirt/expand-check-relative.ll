; RUN: opt -S -passes=wholeprogramdevirt %s | FileCheck %s

; Test that we correctly expand the llvm.type.checked.load.relative intrinsic in
; cases where we cannot devirtualize.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant { [2 x i32] } { [2 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vf1 to i64), i64 ptrtoint (ptr @vt1 to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vf2 to i64), i64 ptrtoint (ptr @vt1 to i64)) to i32)
]}, align 8, !type !0, !type !1

!0 = !{i64 0, !"vfunc1.type"}
!1 = !{i64 4, !"vfunc2.type"}


define void @vf1(ptr %this) {
  ret void
}

define void @vf2(ptr %this) {
  ret void
}

; CHECK: define void @call
; CHECK:  [[TT:%.*]] = call i1 @llvm.type.test(ptr [[VT:%.*]], metadata !"vfunc1.type")
; CHECK:  br i1 [[TT]]

; Relative pointer computation at the address of the i32 value to the i32 value
; to get to the pointer value.

; CHECK:  [[T0:%.*]] = getelementptr i8, ptr [[VT]], i32 0
; CHECK:  [[T1:%.*]] = load i32, ptr [[T0]]
; CHECK:  [[T2:%.*]] = sext i32 [[T1]] to i64
; CHECK:  [[T3:%.*]] = ptrtoint ptr [[T0]] to i64
; CHECK:  [[T4:%.*]] = add i64 [[T3]], [[T2]]
; CHECK:  [[F:%.*]] = inttoptr i64 [[T4]] to ptr
; CHECK:  call void [[F]](ptr

define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr %vtable, i32 0, metadata !"vfunc1.type")
  %p = extractvalue {ptr, i1} %pair, 1
  br i1 %p, label %cont, label %trap

cont:
  %fptr = extractvalue {ptr, i1} %pair, 0
  call void %fptr(ptr %obj)
  ret void

trap:
  call void @llvm.trap()
  unreachable
}

declare {ptr, i1} @llvm.type.checked.load.relative(ptr, i32, metadata)
declare void @llvm.trap()
