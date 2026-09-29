; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

%vtTy = type { [2 x ptr], [2 x ptr] }

@vt = constant %vtTy { [2 x ptr] [ptr null, ptr @vf1], [2 x ptr] [ptr null, ptr @vf2] }, !type !0, !type !1

define void @vf1(ptr %this) {
  ret void
}

define void @vf2(ptr %this) {
  ret void
}

; CHECK: define void @call1
define void @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call void @vf1(
  call void %fptr(ptr %obj)
  ret void
}

; CHECK: define void @call2
define void @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call void @vf2(
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 8, !"typeid1"}
!1 = !{i32 24, !"typeid2"}
