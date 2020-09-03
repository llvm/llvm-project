; RUN: opt -S -wholeprogramdevirt -whole-program-visibility %s | FileCheck %s
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: private constant { [8 x i8], [1 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\01", [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf1 to i8*)], [0 x i8] zeroinitializer }, !type [[T8:![0-9]+]]
; CHECK: private constant { [8 x i8], [1 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\02", [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf2 to i8*)], [0 x i8] zeroinitializer }, !type [[T8]]
; CHECK: private constant { [8 x i8], [1 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\01", [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf4 to i8*)], [0 x i8] zeroinitializer }, !type [[T8]]
; CHECK: private constant { [8 x i8], [1 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\02", [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf8 to i8*)], [0 x i8] zeroinitializer }, !type [[T8]]

@vt1 = constant [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf1 to i8*)], !type !0
@vt2 = constant [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf2 to i8*)], !type !0
@vt4 = constant [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf4 to i8*)], !type !0
@vt8 = constant [1 x i8*] [i8* bitcast (i1 (i8*, i32)* @vf8 to i8*)], !type !0

define i1 @vf1(i8* %this, i32 %arg) readnone {
  %and = and i32 %arg, 1
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

define i1 @vf2(i8* %this, i32 %arg) readnone {
  %and = and i32 %arg, 2
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

define i1 @vf4(i8* %this, i32 %arg) readnone {
  %and = and i32 %arg, 4
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

define i1 @vf8(i8* %this, i32 %arg) readnone {
  %and = and i32 %arg, 8
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

; CHECK: define i1 @call1
define i1 @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*, i32)*
  ; CHECK: getelementptr {{.*}} -1
  ; CHECK: and {{.*}}, 1
  %result = call i1 %fptr_casted(i8* %obj, i32 5)
  ret i1 %result
}

; CHECK: define i1 @call2
define i1 @call2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*, i32)*
  ; CHECK: getelementptr {{.*}} -1
  ; CHECK: and {{.*}}, 2
  %result = call i1 %fptr_casted(i8* %obj, i32 10)
  ret i1 %result
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

; CHECK: [[T8]] = !{i32 8, !"typeid"}
!0 = !{i32 0, !"typeid"}
