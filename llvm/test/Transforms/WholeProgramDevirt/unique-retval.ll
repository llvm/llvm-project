; RUN: opt -S -wholeprogramdevirt %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf0 to i8*)]
@vt2 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf0 to i8*)]
@vt3 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf1 to i8*)]
@vt4 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf1 to i8*)]

define i1 @vf0(i8* %this) readnone {
  ret i1 0
}

define i1 @vf1(i8* %this) readnone {
  ret i1 1
}

; CHECK: define i1 @call1
define i1 @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  ; CHECK: [[VT1:%[^ ]*]] = bitcast [1 x i8*]* {{.*}} to i8*
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.bitset.test(i8* %vtablei8, metadata !"bitset1")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*)*
  ; CHECK: [[RES1:%[^ ]*]] = icmp eq i8* [[VT1]], bitcast ([1 x i8*]* @vt3 to i8*)
  %result = call i1 %fptr_casted(i8* %obj)
  ; CHECK: ret i1 [[RES1]]
  ret i1 %result
}

; CHECK: define i1 @call2
define i1 @call2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  ; CHECK: [[VT2:%[^ ]*]] = bitcast [1 x i8*]* {{.*}} to i8*
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.bitset.test(i8* %vtablei8, metadata !"bitset2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*)*
  ; CHECK: [[RES1:%[^ ]*]] = icmp ne i8* [[VT1]], bitcast ([1 x i8*]* @vt2 to i8*)
  %result = call i1 %fptr_casted(i8* %obj)
  ret i1 %result
}

declare i1 @llvm.bitset.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{!"bitset1", [1 x i8*]* @vt1, i32 0}
!1 = !{!"bitset1", [1 x i8*]* @vt2, i32 0}
!2 = !{!"bitset1", [1 x i8*]* @vt3, i32 0}
!3 = !{!"bitset2", [1 x i8*]* @vt2, i32 0}
!4 = !{!"bitset2", [1 x i8*]* @vt3, i32 0}
!5 = !{!"bitset2", [1 x i8*]* @vt4, i32 0}
!llvm.bitsets = !{!0, !1, !2, !3, !4, !5}
