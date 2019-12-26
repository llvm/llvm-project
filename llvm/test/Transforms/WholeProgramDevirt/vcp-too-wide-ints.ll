; RUN: opt -S -wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x i8*] [i8* bitcast (i64 (i8*, i128)* @vf1 to i8*)], !type !0
@vt2 = constant [1 x i8*] [i8* bitcast (i64 (i8*, i128)* @vf2 to i8*)], !type !0
@vt3 = constant [1 x i8*] [i8* bitcast (i128 (i8*, i64)* @vf3 to i8*)], !type !1
@vt4 = constant [1 x i8*] [i8* bitcast (i128 (i8*, i64)* @vf4 to i8*)], !type !1

define i64 @vf1(i8* %this, i128 %arg) readnone {
  %argtrunc = trunc i128 %arg to i64
  ret i64 %argtrunc
}

define i64 @vf2(i8* %this, i128 %arg) readnone {
  %argtrunc = trunc i128 %arg to i64
  ret i64 %argtrunc
}

define i128 @vf3(i8* %this, i64 %arg) readnone {
  %argzext = zext i64 %arg to i128
  ret i128 %argzext
}

define i128 @vf4(i8* %this, i64 %arg) readnone {
  %argzext = zext i64 %arg to i128
  ret i128 %argzext
}

; CHECK: define i64 @call1
define i64 @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i64 (i8*, i128)*
  ; CHECK: call i64 %
  %result = call i64 %fptr_casted(i8* %obj, i128 1)
  ret i64 %result
}

; CHECK: define i128 @call2
define i128 @call2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i128 (i8*, i64)*
  ; CHECK: call i128 %
  %result = call i128 %fptr_casted(i8* %obj, i64 1)
  ret i128 %result
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
