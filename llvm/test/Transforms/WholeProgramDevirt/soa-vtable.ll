; RUN: opt -S -wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

%vtTy = type { [2 x void (i8*)*], [2 x void (i8*)*] }

@vt = constant %vtTy { [2 x void (i8*)*] [void (i8*)* null, void (i8*)* @vf1], [2 x void (i8*)*] [void (i8*)* null, void (i8*)* @vf2] }, !type !0, !type !1

define void @vf1(i8* %this) {
  ret void
}

define void @vf2(i8* %this) {
  ret void
}

; CHECK: define void @call1
define void @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void @vf1(
  call void %fptr_casted(i8* %obj)
  ret void
}

; CHECK: define void @call2
define void @call2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void @vf2(
  call void %fptr_casted(i8* %obj)
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 8, !"typeid1"}
!1 = !{i32 24, !"typeid2"}
