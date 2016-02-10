; RUN: opt -S -wholeprogramdevirt %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = global [1 x i8*] [i8* bitcast (i128 (i8*, i128)* @vf1 to i8*)]
@vt2 = global [1 x i8*] [i8* bitcast (i128 (i8*, i128)* @vf2 to i8*)]

define i128 @vf1(i8* %this, i128 %arg) readnone {
  ret i128 %arg
}

define i128 @vf2(i8* %this, i128 %arg) readnone {
  ret i128 %arg
}

; CHECK: define i128 @call
define i128 @call(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.bitset.test(i8* %vtablei8, metadata !"bitset")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i128 (i8*, i128)*
  ; CHECK: call i128 %
  %result = call i128 %fptr_casted(i8* %obj, i128 1)
  ret i128 %result
}

declare i1 @llvm.bitset.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{!"bitset", [1 x i8*]* @vt1, i32 0}
!1 = !{!"bitset", [1 x i8*]* @vt2, i32 0}
!llvm.bitsets = !{!0}
