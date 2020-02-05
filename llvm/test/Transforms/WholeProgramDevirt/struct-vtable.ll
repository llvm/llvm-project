; RUN: opt -S -wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

%vtTy = type { void (i8*)* }

@vt = constant %vtTy { void (i8*)* @vf }, !type !0

define void @vf(i8* %this) {
  ret void
}

; CHECK: define void @call
define void @call(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void @vf(
  call void %fptr_casted(i8* %obj)
  ret void
}

; CHECK: define void @call_oob
define void @call_oob(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 4
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void %
  call void %fptr_casted(i8* %obj)
  ret void
}

; CHECK: define void @call_unaligned
define void @call_unaligned(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, i8* %vtablei8, i32 1
  %fptrptr_casted = bitcast i8* %fptrptr to i8**
  %fptr = load i8*, i8** %fptrptr_casted
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void %
  call void %fptr_casted(i8* %obj)
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
