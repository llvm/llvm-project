; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

; Test that we correctly handle function type mismatches in argument counts
; and bitwidths. We handle an argument count mismatch by refusing
; to optimize. For bitwidth mismatches, we allow the optimization in order
; to simplify the implementation. This is legal because the bitwidth mismatch
; gives the call undefined behavior.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x ptr] [ptr @vf1], !type !0
@vt2 = constant [1 x ptr] [ptr @vf2], !type !0

define i32 @vf1(ptr %this, i32 %arg) readnone {
  ret i32 %arg
}

define i32 @vf2(ptr %this, i32 %arg) readnone {
  ret i32 %arg
}

; CHECK: define i32 @bad_arg_type
define i32 @bad_arg_type(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i64 1)
  ; CHECK: ret i32 1
  ret i32 %result
}

; CHECK: define i32 @bad_arg_count
define i32 @bad_arg_count(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call i32 %
  %result = call i32 %fptr(ptr %obj, i64 1, i64 2)
  ret i32 %result
}

; CHECK: define i64 @bad_return_type
define i64 @bad_return_type(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i64 %fptr(ptr %obj, i32 1)
  ; CHECK: ret i64 1
  ret i64 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
