; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [2 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf1a to i8*), i8* bitcast (i32 (i8*, i32)* @vf1b to i8*)], !type !0
@vt2 = constant [2 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2a to i8*), i8* bitcast (i32 (i8*, i32)* @vf2b to i8*)], !type !0

@sink = external global i32

define i32 @vf1a(i8* %this, i32 %arg) {
  store i32 %arg, i32* @sink
  ret i32 %arg
}

define i32 @vf2a(i8* %this, i32 %arg) {
  store i32 %arg, i32* @sink
  ret i32 %arg
}

define i32 @vf1b(i8* %this, i32 %arg) {
  ret i32 %arg
}

define i32 @vf2b(i8* %this, i32 %arg) {
  ret i32 %arg
}

; Test that we don't apply VCP if the virtual function body accesses memory,
; even if the function returns a constant.

; CHECK: define i32 @call1
define i32 @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*, i32)*
  ; CHECK: call i32 %
  %result = call i32 %fptr_casted(i8* %obj, i32 1)
  ret i32 %result
}

; Test that we can apply VCP regardless of the function attributes by analyzing
; the function body itself.

; CHECK: define i32 @call2
define i32 @call2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 1
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*, i32)*
  %result = call i32 %fptr_casted(i8* %obj, i32 1)
  ; CHECK: ret i32 1
  ret i32 %result
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
