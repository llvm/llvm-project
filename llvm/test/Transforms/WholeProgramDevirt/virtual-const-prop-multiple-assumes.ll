; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt2 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i1,
ptr @vf2i32
], !type !0

define i1 @vf0i1(ptr %this) readnone {
  ret i1 0
}

define i1 @vf1i1(ptr %this) readnone {
  ret i1 1
}

define i32 @vf2i32(ptr %this) readnone {
  ret i32 2
}

; CHECK: define i1 @call1(
define i1 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %p2 = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p2)
  %fptr = load ptr, ptr %vtable
  %result = call i1 %fptr(ptr %obj)
  ret i1 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}