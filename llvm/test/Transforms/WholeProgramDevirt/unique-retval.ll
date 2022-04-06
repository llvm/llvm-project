; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt -stats %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: {{.*}} unique-ret-val: devirtualized a call to vf0
; CHECK: remark: {{.*}} unique-ret-val: devirtualized a call to vf0
; CHECK: remark: {{.*}} devirtualized vf0
; CHECK: remark: {{.*}} devirtualized vf1

@vt1 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf0 to i8*)], !type !0
@vt2 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf0 to i8*)], !type !0, !type !1
@vt3 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf1 to i8*)], !type !0, !type !1
@vt4 = constant [1 x i8*] [i8* bitcast (i1 (i8*)* @vf1 to i8*)], !type !1

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
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*)*
  ; CHECK: [[RES1:%[^ ]*]] = icmp eq [1 x i8*]* %vtable, @vt3
  %result = call i1 %fptr_casted(i8* %obj)
  ; CHECK: ret i1 [[RES1]]
  ret i1 %result
}

; CHECK: define i32 @call2
define i32 @call2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  ; Intentional type mismatch to test zero extend.
  %fptr_casted = bitcast i8* %fptr to i32 (i8*)*
  ; CHECK: [[RES2:%[^ ]*]] = icmp ne [1 x i8*]* %vtable, @vt2
  %result = call i32 %fptr_casted(i8* %obj)
  ; CHECK: [[ZEXT2:%[^ ]*]] = zext i1 [[RES2]] to i32
  ; CHECK: ret i32 [[ZEXT2:%[^ ]*]]
  ret i32 %result
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}

; CHECK: 2 wholeprogramdevirt - Number of whole program devirtualization targets
; CHECK: 2 wholeprogramdevirt - Number of unique return value optimizations
