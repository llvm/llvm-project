; RUN: opt -S -wholeprogramdevirt -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: <unknown>:0:0: virtual-const-prop: devirtualized a call to vf1i32
; CHECK: remark: <unknown>:0:0: virtual-const-prop-1-bit: devirtualized a call to vf1i1
; CHECK: remark: <unknown>:0:0: virtual-const-prop-1-bit: devirtualized a call to vf0i1
; CHECK: remark: <unknown>:0:0: devirtualized vf0i1
; CHECK: remark: <unknown>:0:0: devirtualized vf1i1
; CHECK: remark: <unknown>:0:0: devirtualized vf1i32
; CHECK: remark: <unknown>:0:0: devirtualized vf2i32
; CHECK: remark: <unknown>:0:0: devirtualized vf3i32
; CHECK: remark: <unknown>:0:0: devirtualized vf4i32
; CHECK-NOT: devirtualized

; CHECK: [[VT1DATA:@[^ ]*]] = private constant { [8 x i8], [3 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\01\01\00\00\00", [3 x i8*] [i8* bitcast (i1 (i8*)* @vf0i1 to i8*), i8* bitcast (i1 (i8*)* @vf1i1 to i8*), i8* bitcast (i32 (i8*)* @vf1i32 to i8*)], [0 x i8] zeroinitializer }, section "vt1sec", !type [[T8:![0-9]+]]
@vt1 = constant [3 x i8*] [
i8* bitcast (i1 (i8*)* @vf0i1 to i8*),
i8* bitcast (i1 (i8*)* @vf1i1 to i8*),
i8* bitcast (i32 (i8*)* @vf1i32 to i8*)
], section "vt1sec", !type !0

; CHECK: [[VT2DATA:@[^ ]*]] = private constant { [8 x i8], [3 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\02\02\00\00\00", [3 x i8*] [i8* bitcast (i1 (i8*)* @vf1i1 to i8*), i8* bitcast (i1 (i8*)* @vf0i1 to i8*), i8* bitcast (i32 (i8*)* @vf2i32 to i8*)], [0 x i8] zeroinitializer }, !type [[T8]]
@vt2 = constant [3 x i8*] [
i8* bitcast (i1 (i8*)* @vf1i1 to i8*),
i8* bitcast (i1 (i8*)* @vf0i1 to i8*),
i8* bitcast (i32 (i8*)* @vf2i32 to i8*)
], !type !0

; CHECK: [[VT3DATA:@[^ ]*]] = private constant { [8 x i8], [3 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\01\03\00\00\00", [3 x i8*] [i8* bitcast (i1 (i8*)* @vf0i1 to i8*), i8* bitcast (i1 (i8*)* @vf1i1 to i8*), i8* bitcast (i32 (i8*)* @vf3i32 to i8*)], [0 x i8] zeroinitializer }, !type [[T8]]
@vt3 = constant [3 x i8*] [
i8* bitcast (i1 (i8*)* @vf0i1 to i8*),
i8* bitcast (i1 (i8*)* @vf1i1 to i8*),
i8* bitcast (i32 (i8*)* @vf3i32 to i8*)
], !type !0

; CHECK: [[VT4DATA:@[^ ]*]] = private constant { [8 x i8], [3 x i8*], [0 x i8] } { [8 x i8] c"\00\00\00\02\04\00\00\00", [3 x i8*] [i8* bitcast (i1 (i8*)* @vf1i1 to i8*), i8* bitcast (i1 (i8*)* @vf0i1 to i8*), i8* bitcast (i32 (i8*)* @vf4i32 to i8*)], [0 x i8] zeroinitializer }, !type [[T8]]
@vt4 = constant [3 x i8*] [
i8* bitcast (i1 (i8*)* @vf1i1 to i8*),
i8* bitcast (i1 (i8*)* @vf0i1 to i8*),
i8* bitcast (i32 (i8*)* @vf4i32 to i8*)
], !type !0

; CHECK: @vt5 = {{.*}}, !type [[T0:![0-9]+]]
@vt5 = constant [3 x i8*] [
i8* bitcast (void ()* @__cxa_pure_virtual to i8*),
i8* bitcast (void ()* @__cxa_pure_virtual to i8*),
i8* bitcast (void ()* @__cxa_pure_virtual to i8*)
], !type !0

; CHECK: @vt1 = alias [3 x i8*], getelementptr inbounds ({ [8 x i8], [3 x i8*], [0 x i8] }, { [8 x i8], [3 x i8*], [0 x i8] }* [[VT1DATA]], i32 0, i32 1)
; CHECK: @vt2 = alias [3 x i8*], getelementptr inbounds ({ [8 x i8], [3 x i8*], [0 x i8] }, { [8 x i8], [3 x i8*], [0 x i8] }* [[VT2DATA]], i32 0, i32 1)
; CHECK: @vt3 = alias [3 x i8*], getelementptr inbounds ({ [8 x i8], [3 x i8*], [0 x i8] }, { [8 x i8], [3 x i8*], [0 x i8] }* [[VT3DATA]], i32 0, i32 1)
; CHECK: @vt4 = alias [3 x i8*], getelementptr inbounds ({ [8 x i8], [3 x i8*], [0 x i8] }, { [8 x i8], [3 x i8*], [0 x i8] }* [[VT4DATA]], i32 0, i32 1)

define i1 @vf0i1(i8* %this) readnone {
  ret i1 0
}

define i1 @vf1i1(i8* %this) readnone {
  ret i1 1
}

define i32 @vf1i32(i8* %this) readnone {
  ret i32 1
}

define i32 @vf2i32(i8* %this) readnone {
  ret i32 2
}

define i32 @vf3i32(i8* %this) readnone {
  ret i32 3
}

define i32 @vf4i32(i8* %this) readnone {
  ret i32 4
}

; CHECK: define i1 @call1(
define i1 @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [3 x i8*]**
  %vtable = load [3 x i8*]*, [3 x i8*]** %vtableptr
  ; CHECK: [[VT1:%[^ ]*]] = bitcast [3 x i8*]* {{.*}} to i8*
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtablei8, i32 0, metadata !"typeid")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to i1 (i8*)*
  ; CHECK: [[VTGEP1:%[^ ]*]] = getelementptr i8, i8* [[VT1]], i32 -5
  ; CHECK: [[VTLOAD1:%[^ ]*]] = load i8, i8* [[VTGEP1]]
  ; CHECK: [[VTAND1:%[^ ]*]] = and i8 [[VTLOAD1]], 2
  ; CHECK: [[VTCMP1:%[^ ]*]] = icmp ne i8 [[VTAND1]], 0
  %result = call i1 %fptr_casted(i8* %obj)
  ; CHECK: [[AND1:%[^ ]*]] = and i1 [[VTCMP1]], true
  %p = extractvalue {i8*, i1} %pair, 1
  %and = and i1 %result, %p
  ; CHECK: ret i1 [[AND1]]
  ret i1 %and
}

; CHECK: define i1 @call2(
define i1 @call2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [3 x i8*]**
  %vtable = load [3 x i8*]*, [3 x i8*]** %vtableptr
  ; CHECK: [[VT2:%[^ ]*]] = bitcast [3 x i8*]* {{.*}} to i8*
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtablei8, i32 8, metadata !"typeid")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to i1 (i8*)*
  ; CHECK: [[VTGEP2:%[^ ]*]] = getelementptr i8, i8* [[VT2]], i32 -5
  ; CHECK: [[VTLOAD2:%[^ ]*]] = load i8, i8* [[VTGEP2]]
  ; CHECK: [[VTAND2:%[^ ]*]] = and i8 [[VTLOAD2]], 1
  ; CHECK: [[VTCMP2:%[^ ]*]] = icmp ne i8 [[VTAND2]], 0
  %result = call i1 %fptr_casted(i8* %obj)
  ; CHECK: [[AND2:%[^ ]*]] = and i1 [[VTCMP2]], true
  %p = extractvalue {i8*, i1} %pair, 1
  %and = and i1 %result, %p
  ; CHECK: ret i1 [[AND2]]
  ret i1 %and
}

; CHECK: define i32 @call3(
define i32 @call3(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [3 x i8*]**
  %vtable = load [3 x i8*]*, [3 x i8*]** %vtableptr
  ; CHECK: [[VT3:%[^ ]*]] = bitcast [3 x i8*]* {{.*}} to i8*
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtablei8, i32 16, metadata !"typeid")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to i32 (i8*)*
  ; CHECK: [[VTGEP3:%[^ ]*]] = getelementptr i8, i8* [[VT3]], i32 -4
  ; CHECK: [[VTBC3:%[^ ]*]] = bitcast i8* [[VTGEP3]] to i32*
  ; CHECK: [[VTLOAD3:%[^ ]*]] = load i32, i32* [[VTBC3]]
  %result = call i32 %fptr_casted(i8* %obj)
  ; CHECK: ret i32 [[VTLOAD3]]
  ret i32 %result
}

declare {i8*, i1} @llvm.type.checked.load(i8*, i32, metadata)
declare void @llvm.assume(i1)
declare void @__cxa_pure_virtual()

; CHECK: [[T8]] = !{i32 8, !"typeid"}
; CHECK: [[T0]] = !{i32 0, !"typeid"}

!0 = !{i32 0, !"typeid"}
