; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x ptr] [ptr @vf], !type !0
@vt2 = constant [1 x ptr] [ptr @vf], !type !0

define void @vf(ptr %this) {
  ret void
}

; CHECK: define void @call
define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %p2 = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p2)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call void @vf(
  call void %fptr(ptr %obj)
  ret void
}

declare ptr @llvm.load.relative.i32(ptr, i32)

@vt3 = private unnamed_addr constant [1 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf to i64), i64 ptrtoint (ptr @vt3 to i64)) to i32)
], align 4, !type !1

; CHECK: define void @call2
define void @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %p2 = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p2)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  ; CHECK: call void @vf(
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
!1 = !{i32 0, !"typeid2"}
