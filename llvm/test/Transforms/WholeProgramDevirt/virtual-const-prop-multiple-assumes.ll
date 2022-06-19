; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt2 = constant [3 x i8*] [
i8* bitcast (i1 (i8*)* @vf1i1 to i8*),
i8* bitcast (i1 (i8*)* @vf0i1 to i8*),
i8* bitcast (i32 (i8*)* @vf2i32 to i8*)
], !type !0

define i1 @vf0i1(i8* %this) readnone {
  ret i1 0
}

define i1 @vf1i1(i8* %this) readnone {
  ret i1 1
}

define i32 @vf2i32(i8* %this) readnone {
  ret i32 2
}

; CHECK: define i1 @call1(
define i1 @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [3 x i8*]**
  %vtable = load [3 x i8*]*, [3 x i8*]** %vtableptr
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %p2 = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p2)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*)*
  %result = call i1 %fptr_casted(i8* %obj)
  ret i1 %result
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}