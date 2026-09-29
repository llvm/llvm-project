; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

; Test that we correctly expand the llvm.type.checked.load intrinsic in cases
; where we cannot devirtualize.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x ptr] [ptr @vf1], !type !0
@vt2 = constant [1 x ptr] [ptr @vf2], !type !0

define void @vf1(ptr %this) {
  ret void
}

define void @vf2(ptr %this) {
  ret void
}

; CHECK: define void @call
define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid")
  %p = extractvalue {ptr, i1} %pair, 1
  ; CHECK: [[TT:%[^ ]*]] = call i1 @llvm.type.test(ptr [[VT:%[^,]*]], metadata !"typeid")
  ; CHECK: br i1 [[TT]],
  br i1 %p, label %cont, label %trap

cont:
  ; CHECK: [[GEP:%[^ ]*]] = getelementptr i8, ptr [[VT]], i32 0
  ; CHECK: [[LOAD:%[^ ]*]] = load ptr, ptr [[GEP]]
  ; CHECK: call void [[LOAD]]
  %fptr = extractvalue {ptr, i1} %pair, 0
  call void %fptr(ptr %obj)
  ret void

trap:
  call void @llvm.trap()
  unreachable
}

; CHECK: define { ptr, i1 } @ret
define {ptr, i1} @ret(ptr %vtablei8) {
  ; CHECK: [[GEP2:%[^ ]*]] = getelementptr i8, ptr [[VT2:%[^,]*]], i32 1
  ; CHECK: [[LOAD2:%[^ ]*]] = load ptr, ptr [[GEP2]]
  ; CHECK: [[TT2:%[^ ]*]] = call i1 @llvm.type.test(ptr %vtablei8, metadata !"typeid")
  ; CHECK: [[I1:%[^ ]*]] = insertvalue { ptr, i1 } poison, ptr [[LOAD2]], 0
  ; CHECK: [[I2:%[^ ]*]] = insertvalue { ptr, i1 } [[I1]], i1 [[TT2]], 1
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtablei8, i32 1, metadata !"typeid")
  ; CHECK: ret { ptr, i1 } [[I2]]
  ret {ptr, i1} %pair
}

declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare void @llvm.trap()

!0 = !{i32 0, !"typeid"}
