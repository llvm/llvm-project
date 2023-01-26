; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

; CHECK-NOT: devirtualized call

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt = global [1 x ptr] [ptr @vf], !type !0

define void @vf(ptr %this) {
  ret void
}

; CHECK: define void @call
define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call void %
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
