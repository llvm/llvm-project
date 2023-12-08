; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

%vtTy = type { ptr }

@vt = constant %vtTy { ptr @vf }, !type !0

define void @vf(ptr %this) {
  ret void
}

; CHECK: define void @call
define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call void @vf(
  call void %fptr(ptr %obj)
  ret void
}

; CHECK: define void @call_oob
define void @call_oob(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x ptr], ptr %vtable, i32 0, i32 4
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: call void %
  call void %fptr(ptr %obj)
  ret void
}

; CHECK: define void @call_unaligned
define void @call_unaligned(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, ptr %vtable, i32 1
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: call void %
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
