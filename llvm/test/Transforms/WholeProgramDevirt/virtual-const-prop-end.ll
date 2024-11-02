; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: [[VT1DATA:@[^ ]*]] = private constant { [0 x i8], [4 x ptr], [5 x i8] } { [0 x i8] zeroinitializer, [4 x ptr] [ptr null, ptr @vf0i1, ptr @vf1i1, ptr @vf1i32], [5 x i8] c"\02\01\00\00\00" }, !type [[T8:![0-9]+]]
@vt1 = constant [4 x ptr] [
ptr null,
ptr @vf0i1,
ptr @vf1i1,
ptr @vf1i32
], !type !1

; CHECK: [[VT2DATA:@[^ ]*]] = private constant { [0 x i8], [3 x ptr], [5 x i8] } { [0 x i8] zeroinitializer, [3 x ptr] [ptr @vf1i1, ptr @vf0i1, ptr @vf2i32], [5 x i8] c"\01\02\00\00\00" }, !type [[T0:![0-9]+]]
@vt2 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i1,
ptr @vf2i32
], !type !0

; CHECK: [[VT3DATA:@[^ ]*]] = private constant { [0 x i8], [4 x ptr], [5 x i8] } { [0 x i8] zeroinitializer, [4 x ptr] [ptr null, ptr @vf0i1, ptr @vf1i1, ptr @vf3i32], [5 x i8] c"\02\03\00\00\00" }, !type [[T8]]
@vt3 = constant [4 x ptr] [
ptr null,
ptr @vf0i1,
ptr @vf1i1,
ptr @vf3i32
], !type !1

; CHECK: [[VT4DATA:@[^ ]*]] = private constant { [0 x i8], [3 x ptr], [5 x i8] } { [0 x i8] zeroinitializer, [3 x ptr] [ptr @vf1i1, ptr @vf0i1, ptr @vf4i32], [5 x i8] c"\01\04\00\00\00" }, !type [[T0]]
@vt4 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i1,
ptr @vf4i32
], !type !0

; CHECK: @vt1 = alias [4 x ptr], getelementptr inbounds ({ [0 x i8], [4 x ptr], [5 x i8] }, ptr [[VT1DATA]], i32 0, i32 1)
; CHECK: @vt2 = alias [3 x ptr], getelementptr inbounds ({ [0 x i8], [3 x ptr], [5 x i8] }, ptr [[VT2DATA]], i32 0, i32 1)
; CHECK: @vt3 = alias [4 x ptr], getelementptr inbounds ({ [0 x i8], [4 x ptr], [5 x i8] }, ptr [[VT3DATA]], i32 0, i32 1)
; CHECK: @vt4 = alias [3 x ptr], getelementptr inbounds ({ [0 x i8], [3 x ptr], [5 x i8] }, ptr [[VT4DATA]], i32 0, i32 1)

define i1 @vf0i1(ptr %this) readnone {
  ret i1 0
}

define i1 @vf1i1(ptr %this) readnone {
  ret i1 1
}

define i32 @vf1i32(ptr %this) readnone {
  ret i32 1
}

define i32 @vf2i32(ptr %this) readnone {
  ret i32 2
}

define i32 @vf3i32(ptr %this) readnone {
  ret i32 3
}

define i32 @vf4i32(ptr %this) readnone {
  ret i32 4
}

; CHECK: define i1 @call1(
define i1 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: [[VTGEP1:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 24
  ; CHECK: [[VTLOAD1:%[^ ]*]] = load i8, ptr [[VTGEP1]]
  ; CHECK: [[VTAND1:%[^ ]*]] = and i8 [[VTLOAD1]], 1
  ; CHECK: [[VTCMP1:%[^ ]*]] = icmp ne i8 [[VTAND1]], 0
  %result = call i1 %fptr(ptr %obj)
  ; CHECK: ret i1 [[VTCMP1]]
  ret i1 %result
}

; CHECK: define i1 @call2(
define i1 @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 1
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: [[VTGEP2:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 24
  ; CHECK: [[VTLOAD2:%[^ ]*]] = load i8, ptr [[VTGEP2]]
  ; CHECK: [[VTAND2:%[^ ]*]] = and i8 [[VTLOAD2]], 2
  ; CHECK: [[VTCMP2:%[^ ]*]] = icmp ne i8 [[VTAND2]], 0
  %result = call i1 %fptr(ptr %obj)
  ; CHECK: ret i1 [[VTCMP2]]
  ret i1 %result
}

; CHECK: define i32 @call3(
define i32 @call3(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 2
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: [[VTGEP3:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 25
  ; CHECK: [[VTLOAD3:%[^ ]*]] = load i32, ptr [[VTGEP3]]
  %result = call i32 %fptr(ptr %obj)
  ; CHECK: ret i32 [[VTLOAD3]]
  ret i32 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

; CHECK: [[T8]] = !{i32 8, !"typeid"}
; CHECK: [[T0]] = !{i32 0, !"typeid"}

!0 = !{i32 0, !"typeid"}
!1 = !{i32 8, !"typeid"}
