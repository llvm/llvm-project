; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [2 x i8*] [i8* zeroinitializer, i8* bitcast (void (i8*)* @vf to i8*)], !type !0
@vt2 = constant i8* bitcast (void (i8*)* @vf to i8*), !type !1

define void @vf(i8* %this) {
  ret void
}

; CHECK: define void @unaligned1
define void @unaligned1(i8* %obj) {
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

; CHECK: define void @unaligned2
define void @unaligned2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, i8* %vtablei8, i32 1
  %fptrptr_casted = bitcast i8* %fptrptr to i8**
  %fptr = load i8*, i8** %fptrptr_casted
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void %
  call void %fptr_casted(i8* %obj)
  ret void
}

; CHECK: define void @outofbounds
define void @outofbounds(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, i8* %vtablei8, i32 16
  %fptrptr_casted = bitcast i8* %fptrptr to i8**
  %fptr = load i8*, i8** %fptrptr_casted
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void %
  call void %fptr_casted(i8* %obj)
  ret void
}

; CHECK: define void @nonfunction
define void @nonfunction(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8, i8* %vtablei8, i32 0
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
!1 = !{i32 0, !"typeid2"}
