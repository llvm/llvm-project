; RUN: opt -S -wholeprogramdevirt %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = global [1 x i8*] [i8* bitcast (i32 ()* @vf1 to i8*)]
@vt2 = global [1 x i8*] [i8* bitcast (i32 ()* @vf2 to i8*)]

define i32 @vf1() readnone {
  ret i32 1
}

define i32 @vf2() readnone {
  ret i32 2
}

; CHECK: define i32 @call
define i32 @call(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.bitset.test(i8* %vtablei8, metadata !"bitset")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 ()*
  ; CHECK: call i32 %
  %result = call i32 %fptr_casted()
  ret i32 %result
}

declare i1 @llvm.bitset.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{!"bitset", [1 x i8*]* @vt1, i32 0}
!1 = !{!"bitset", [1 x i8*]* @vt2, i32 0}
!llvm.bitsets = !{!0}
