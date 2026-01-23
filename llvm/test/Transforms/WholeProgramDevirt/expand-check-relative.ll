; RUN: opt -S -passes=wholeprogramdevirt %s | FileCheck %s

; Test that we correctly expand the llvm.type.checked.load.relative intrinsic in
; cases where we cannot devirtualize.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant { [2 x i32] } { [2 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vf1 to i64), i64 ptrtoint (ptr @vt1 to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @vf2 to i64), i64 ptrtoint (ptr @vt1 to i64)) to i32)
]}, align 8, !type !0

!0 = !{i64 0, !"vtable"}

define void @vf1(ptr %this) {
  ret void
}

define void @vf2(ptr %this) {
  ret void
}

; CHECK: define void @call_vf1
; CHECK:  [[TT:%.*]] = call i1 @llvm.type.test(ptr [[VT:%.*]], metadata !"vtable")
; CHECK:  br i1 [[TT]]

; Relative pointer computation at the vtable to the i32 value
; to get to the pointer value.

; CHECK:  [[F:%.*]] = call ptr @llvm.load.relative.i32(ptr [[VT]], i32 0)
; CHECK:  call void [[F]](ptr

define void @call_vf1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr %vtable, i32 0, metadata !"vtable")
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

; CHECK: define void @call_vf2
; CHECK:  [[TT:%.*]] = call i1 @llvm.type.test(ptr [[VT:%.*]], metadata !"vtable")
; CHECK:  br i1 [[TT]]

; CHECK:  [[F:%.*]] = call ptr @llvm.load.relative.i32(ptr [[VT]], i32 4)
; CHECK:  call void [[F]](ptr

define void @call_vf2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr %vtable, i32 4, metadata !"vtable")
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
