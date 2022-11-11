; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt -stats %s 2>&1 | FileCheck %s
; Skipping vf0i1 is identical to setting public LTO visibility. We don't devirtualize vf0i1 and all other
; virtual call targets.
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt -wholeprogramdevirt-skip=vf0i1 %s 2>&1 | FileCheck %s --check-prefix=SKIP
; We have two set of call targets {vf0i1, vf1i1} and {vf1i32, vf2i32, vf3i32, vf4i32}.
; The command below prevents both of them from devirtualization.
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt -wholeprogramdevirt-skip=vf0i1,vf1i32 %s 2>&1 | FileCheck %s --check-prefix=SKIP-ALL
; Check wildcard
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt -wholeprogramdevirt-skip=vf?i1 %s 2>&1 | FileCheck %s --check-prefix=SKIP

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: <unknown>:0:0: virtual-const-prop-1-bit: devirtualized a call to vf0i1
; CHECK: remark: <unknown>:0:0: virtual-const-prop-1-bit: devirtualized a call to vf1i1
; CHECK: remark: <unknown>:0:0: virtual-const-prop: devirtualized a call to vf1i32
; CHECK: remark: <unknown>:0:0: devirtualized vf0i1
; CHECK: remark: <unknown>:0:0: devirtualized vf1i1
; CHECK: remark: <unknown>:0:0: devirtualized vf1i32
; CHECK: remark: <unknown>:0:0: devirtualized vf2i32
; CHECK: remark: <unknown>:0:0: devirtualized vf3i32
; CHECK: remark: <unknown>:0:0: devirtualized vf4i32
; CHECK-NOT: devirtualized

; SKIP: remark: <unknown>:0:0: virtual-const-prop: devirtualized a call to vf1i32
; SKIP: remark: <unknown>:0:0: devirtualized vf1i32
; SKIP: remark: <unknown>:0:0: devirtualized vf2i32
; SKIP: remark: <unknown>:0:0: devirtualized vf3i32
; SKIP: remark: <unknown>:0:0: devirtualized vf4i32
; SKIP-NOT: devirtualized

; SKIP-ALL-NOT: devirtualized

; CHECK: [[VT1DATA:@[^ ]*]] = private constant { [8 x i8], [3 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\01\00\00\00\02", [3 x ptr] [ptr @vf0i1, ptr @vf1i1, ptr @vf1i32], [0 x i8] zeroinitializer }, section "vt1sec", !type [[T8:![0-9]+]]
@vt1 = constant [3 x ptr] [
ptr @vf0i1,
ptr @vf1i1,
ptr @vf1i32
], section "vt1sec", !type !0

; CHECK: [[VT2DATA:@[^ ]*]] = private constant { [8 x i8], [3 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\02\00\00\00\01", [3 x ptr] [ptr @vf1i1, ptr @vf0i1, ptr @vf2i32], [0 x i8] zeroinitializer }, !type [[T8]]
@vt2 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i1,
ptr @vf2i32
], !type !0

; CHECK: [[VT3DATA:@[^ ]*]] = private constant { [8 x i8], [3 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\03\00\00\00\02", [3 x ptr] [ptr @vf0i1, ptr @vf1i1, ptr @vf3i32], [0 x i8] zeroinitializer }, !type [[T8]]
@vt3 = constant [3 x ptr] [
ptr @vf0i1,
ptr @vf1i1,
ptr @vf3i32
], !type !0

; CHECK: [[VT4DATA:@[^ ]*]] = private constant { [8 x i8], [3 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\04\00\00\00\01", [3 x ptr] [ptr @vf1i1, ptr @vf0i1, ptr @vf4i32], [0 x i8] zeroinitializer }, !type [[T8]]
@vt4 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i1,
ptr @vf4i32
], !type !0

; CHECK: @vt5 = {{.*}}, !type [[T0:![0-9]+]]
@vt5 = constant [3 x ptr] [
ptr @__cxa_pure_virtual,
ptr @__cxa_pure_virtual,
ptr @__cxa_pure_virtual
], !type !0

; CHECK: @vt1 = alias [3 x ptr], getelementptr inbounds ({ [8 x i8], [3 x ptr], [0 x i8] }, ptr [[VT1DATA]], i32 0, i32 1)
; CHECK: @vt2 = alias [3 x ptr], getelementptr inbounds ({ [8 x i8], [3 x ptr], [0 x i8] }, ptr [[VT2DATA]], i32 0, i32 1)
; CHECK: @vt3 = alias [3 x ptr], getelementptr inbounds ({ [8 x i8], [3 x ptr], [0 x i8] }, ptr [[VT3DATA]], i32 0, i32 1)
; CHECK: @vt4 = alias [3 x ptr], getelementptr inbounds ({ [8 x i8], [3 x ptr], [0 x i8] }, ptr [[VT4DATA]], i32 0, i32 1)

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
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  ; CHECK: [[VTGEP1:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -1
  ; CHECK: [[VTLOAD1:%[^ ]*]] = load i8, ptr [[VTGEP1]]
  ; CHECK: [[VTAND1:%[^ ]*]] = and i8 [[VTLOAD1]], 1
  ; CHECK: [[VTCMP1:%[^ ]*]] = icmp ne i8 [[VTAND1]], 0
  %result = call i1 %fptr(ptr %obj)
  ; CHECK: [[AND1:%[^ ]*]] = and i1 [[VTCMP1]], true
  %p = extractvalue {ptr, i1} %pair, 1
  %and = and i1 %result, %p
  ; CHECK: ret i1 [[AND1]]
  ret i1 %and
}

; CHECK: define i1 @call2(
define i1 @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  ; CHECK: [[VTGEP2:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -1
  ; CHECK: [[VTLOAD2:%[^ ]*]] = load i8, ptr [[VTGEP2]]
  ; CHECK: [[VTAND2:%[^ ]*]] = and i8 [[VTLOAD2]], 2
  ; CHECK: [[VTCMP2:%[^ ]*]] = icmp ne i8 [[VTAND2]], 0
  %result = call i1 %fptr(ptr %obj)
  ; CHECK: [[AND2:%[^ ]*]] = and i1 [[VTCMP2]], true
  %p = extractvalue {ptr, i1} %pair, 1
  %and = and i1 %result, %p
  ; CHECK: ret i1 [[AND2]]
  ret i1 %and
}

; CHECK: define i32 @call3(
define i32 @call3(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 16, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  ; CHECK: [[VTGEP3:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -5
  ; CHECK: [[VTLOAD3:%[^ ]*]] = load i32, ptr [[VTGEP3]]
  %result = call i32 %fptr(ptr %obj)
  ; CHECK: ret i32 [[VTLOAD3]]
  ret i32 %result
}

declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare void @llvm.assume(i1)
declare void @__cxa_pure_virtual()

; CHECK: [[T8]] = !{i32 8, !"typeid"}
; CHECK: [[T0]] = !{i32 0, !"typeid"}

!0 = !{i32 0, !"typeid"}

; CHECK: 6 wholeprogramdevirt - Number of whole program devirtualization targets
; CHECK: 1 wholeprogramdevirt - Number of virtual constant propagations
; CHECK: 2 wholeprogramdevirt - Number of 1 bit virtual constant propagations
