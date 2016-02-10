; RUN: opt -S -wholeprogramdevirt %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = global [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf1 to i8*)]
@vt2 = global [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2 to i8*)]

define i32 @vf1(i8* %this, i32 %arg) readnone {
  ret i32 %arg
}

define i32 @vf2(i8* %this, i32 %arg) readnone {
  ret i32 %arg
}

; CHECK: define i32 @bad_arg_type
define i32 @bad_arg_type(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.bitset.test(i8* %vtablei8, metadata !"bitset")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*, i64)*
  ; CHECK: call i32 %
  %result = call i32 %fptr_casted(i8* %obj, i64 1)
  ret i32 %result
}

; CHECK: define i32 @bad_arg_count
define i32 @bad_arg_count(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.bitset.test(i8* %vtablei8, metadata !"bitset")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*, i64, i64)*
  ; CHECK: call i32 %
  %result = call i32 %fptr_casted(i8* %obj, i64 1, i64 2)
  ret i32 %result
}

; CHECK: define i64 @bad_return_type
define i64 @bad_return_type(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.bitset.test(i8* %vtablei8, metadata !"bitset")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i64 (i8*, i32)*
  ; CHECK: call i64 %
  %result = call i64 %fptr_casted(i8* %obj, i32 1)
  ret i64 %result
}

declare i1 @llvm.bitset.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{!"bitset", [1 x i8*]* @vt1, i32 0}
!1 = !{!"bitset", [1 x i8*]* @vt2, i32 0}
!llvm.bitsets = !{!0}
