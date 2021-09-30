; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

; Test that we correctly expand the llvm.type.checked.load intrinsic in cases
; where we cannot devirtualize.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x i8*] [i8* bitcast (void (i8*)* @vf1 to i8*)], !type !0
@vt2 = constant [1 x i8*] [i8* bitcast (void (i8*)* @vf2 to i8*)], !type !0

define void @vf1(i8* %this) {
  ret void
}

define void @vf2(i8* %this) {
  ret void
}

; CHECK: define void @call
define void @call(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtablei8, i32 0, metadata !"typeid")
  %p = extractvalue {i8*, i1} %pair, 1
  ; CHECK: [[TT:%[^ ]*]] = call i1 @llvm.type.test(i8* [[VT:%[^,]*]], metadata !"typeid")
  ; CHECK: br i1 [[TT]],
  br i1 %p, label %cont, label %trap

cont:
  ; CHECK: [[GEP:%[^ ]*]] = getelementptr i8, i8* [[VT]], i32 0
  ; CHECK: [[BC:%[^ ]*]] = bitcast i8* [[GEP]] to i8**
  ; CHECK: [[LOAD:%[^ ]*]] = load i8*, i8** [[BC]]
  ; CHECK: [[FPC:%[^ ]*]] = bitcast i8* [[LOAD]] to void (i8*)*
  ; CHECK: call void [[FPC]]
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  call void %fptr_casted(i8* %obj)
  ret void

trap:
  call void @llvm.trap()
  unreachable
}

; CHECK: define { i8*, i1 } @ret
define {i8*, i1} @ret(i8* %vtablei8) {
  ; CHECK: [[GEP2:%[^ ]*]] = getelementptr i8, i8* [[VT2:%[^,]*]], i32 1
  ; CHECK: [[BC2:%[^ ]*]] = bitcast i8* [[GEP2]] to i8**
  ; CHECK: [[LOAD2:%[^ ]*]] = load i8*, i8** [[BC2]]
  ; CHECK: [[TT2:%[^ ]*]] = call i1 @llvm.type.test(i8* [[VT2]], metadata !"typeid")
  ; CHECK: [[I1:%[^ ]*]] = insertvalue { i8*, i1 } undef, i8* [[LOAD2]], 0
  ; CHECK: [[I2:%[^ ]*]] = insertvalue { i8*, i1 } %5, i1 [[TT2]], 1
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtablei8, i32 1, metadata !"typeid")
  ; CHECK: ret { i8*, i1 } [[I2]]
  ret {i8*, i1} %pair
}

declare {i8*, i1} @llvm.type.checked.load(i8*, i32, metadata)
declare void @llvm.trap()

!0 = !{i32 0, !"typeid"}
