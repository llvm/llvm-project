; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [2 x ptr] [ptr zeroinitializer, ptr @vf], !type !0
@vt2 = constant ptr @vf, !type !1

define void @vf(ptr %this) {
  ret void
}

; CHECK: define void @unaligned1
define void @unaligned1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, ptr %vtable, i32 1
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: call void %
  call void %fptr(ptr %obj)
  ret void
}

; CHECK: define void @unaligned2
define void @unaligned2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, ptr %vtable, i32 1
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: call void %
  call void %fptr(ptr %obj)
  ret void
}

; CHECK: define void @outofbounds
define void @outofbounds(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, ptr %vtable, i32 16
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: call void %
  call void %fptr(ptr %obj)
  ret void
}

; CHECK: define void @nonfunction
define void @nonfunction(ptr %obj) {
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
!1 = !{i32 0, !"typeid2"}
