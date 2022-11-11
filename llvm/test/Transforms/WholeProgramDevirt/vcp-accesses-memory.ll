; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [2 x ptr] [ptr @vf1a, ptr @vf1b], !type !0
@vt2 = constant [2 x ptr] [ptr @vf2a, ptr @vf2b], !type !0

@sink = external global i32

define i32 @vf1a(ptr %this, i32 %arg) {
  store i32 %arg, ptr @sink
  ret i32 %arg
}

define i32 @vf2a(ptr %this, i32 %arg) {
  store i32 %arg, ptr @sink
  ret i32 %arg
}

define i32 @vf1b(ptr %this, i32 %arg) {
  ret i32 %arg
}

define i32 @vf2b(ptr %this, i32 %arg) {
  ret i32 %arg
}

; Test that we don't apply VCP if the virtual function body accesses memory,
; even if the function returns a constant.

; CHECK: define i32 @call1
define i32 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call i32 %
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

; Test that we can apply VCP regardless of the function attributes by analyzing
; the function body itself.

; CHECK: define i32 @call2
define i32 @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x ptr], ptr %vtable, i32 0, i32 1
  %fptr = load ptr, ptr %fptrptr
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; CHECK: ret i32 1
  ret i32 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
