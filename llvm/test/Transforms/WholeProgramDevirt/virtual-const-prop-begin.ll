; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"

;; Note that i16 is used here such that we can ensure all constants for "typeid"
;; can come before the vtable. For this particular file, the intention is to only
;; look at constants placed before the vtable, but with the next change, this
;; would place the original i32s after the vtable due to extra padding needed to
;; preserve alignment. Making them i16s allows them to stay at the beginning of
;; the vtable. There are other tests where there's a mix of constants before and
;; after the vtable but for this file we just want everything before the vtable.
; CHECK: [[VT1DATA:@[^ ]*]] = private constant { [8 x i8], [3 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\00\03\00\00\02", [3 x ptr] [ptr @vf0i1, ptr @vf1i1, ptr @vf1i16], [0 x i8] zeroinitializer }, section "vt1sec", !type [[T8:![0-9]+]]
@vt1 = constant [3 x ptr] [
ptr @vf0i1,
ptr @vf1i1,
ptr @vf1i16
], section "vt1sec", !type !0

; CHECK: [[VT2DATA:@[^ ]*]] = private constant { [8 x i8], [3 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\00\04\00\00\01", [3 x ptr] [ptr @vf1i1, ptr @vf0i1, ptr @vf2i16], [0 x i8] zeroinitializer }, !type [[T8]]
@vt2 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i1,
ptr @vf2i16
], !type !0

; CHECK: [[VT3DATA:@[^ ]*]] = private constant { [4 x i8], [3 x ptr], [0 x i8] } { [4 x i8] c"\05\00\00\02", [3 x ptr] [ptr @vf0i1, ptr @vf1i1, ptr @vf3i16], [0 x i8] zeroinitializer }, align 2, !type [[T5:![0-9]+]]
@vt3 = constant [3 x ptr] [
ptr @vf0i1,
ptr @vf1i1,
ptr @vf3i16
], align 2, !type !0

; CHECK: [[VT4DATA:@[^ ]*]] = private constant { [16 x i8], [3 x ptr], [0 x i8] } { [16 x i8] c"\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\01", [3 x ptr] [ptr @vf1i1, ptr @vf0i1, ptr @vf4i16], [0 x i8] zeroinitializer },  align 16, !type [[T16:![0-9]+]]
@vt4 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i1,
ptr @vf4i16
], align 16, !type !0

; CHECK: @vt5 = {{.*}}, !type [[T0:![0-9]+]]
@vt5 = constant [3 x ptr] [
ptr @__cxa_pure_virtual,
ptr @__cxa_pure_virtual,
ptr @__cxa_pure_virtual
], !type !0

;; Test relative vtables
; CHECK:      [[VT6RELDATA:@[^ ]*]] = private constant { [4 x i8], [3 x i32], [0 x i8] } { [4 x i8] c"\00\00\03\00", [3 x i32] [
; CHECK-SAME:     i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf0i1 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32),
; CHECK-SAME:     i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1i1 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32),
; CHECK-SAME:     i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1i16 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32)
; CHECK-SAME: ], [0 x i8] zeroinitializer }, !type [[TREL:![0-9]+]]
@vt6_rel = constant [3 x i32] [
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf0i1 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1i1 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1i16 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32)
], !type !2

; CHECK:      [[VT7RELDATA:@[^ ]*]] = private constant { [4 x i8], [3 x i32], [0 x i8] } { [4 x i8] c"\00\00\04\00", [3 x i32] [
; CHECK-SAME:     i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1i1 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32),
; CHECK-SAME:     i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf0i1 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32),
; CHECK-SAME:     i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2i16 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32)
; CHECK-SAME: ], [0 x i8] zeroinitializer }, !type [[TREL]]
@vt7_rel = constant [3 x i32] [
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1i1 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf0i1 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2i16 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32)
], !type !2

; CHECK: @vt1 = alias [3 x ptr], getelementptr inbounds ({ [8 x i8], [3 x ptr], [0 x i8] }, ptr [[VT1DATA]], i32 0, i32 1)
; CHECK: @vt2 = alias [3 x ptr], getelementptr inbounds ({ [8 x i8], [3 x ptr], [0 x i8] }, ptr [[VT2DATA]], i32 0, i32 1)
; CHECK: @vt3 = alias [3 x ptr], getelementptr inbounds ({ [4 x i8], [3 x ptr], [0 x i8] }, ptr [[VT3DATA]], i32 0, i32 1)
; CHECK: @vt4 = alias [3 x ptr], getelementptr inbounds ({ [16 x i8], [3 x ptr], [0 x i8] }, ptr [[VT4DATA]], i32 0, i32 1)
; CHECK: @vt6_rel = alias [3 x i32], getelementptr inbounds ({ [4 x i8], [3 x i32], [0 x i8] }, ptr [[VT6RELDATA]], i32 0, i32 1)
; CHECK: @vt7_rel = alias [3 x i32], getelementptr inbounds ({ [4 x i8], [3 x i32], [0 x i8] }, ptr [[VT7RELDATA]], i32 0, i32 1)

define i1 @vf0i1(ptr %this) readnone {
  ret i1 0
}

define i1 @vf1i1(ptr %this) readnone {
  ret i1 1
}

define i16 @vf1i16(ptr %this) readnone {
  ret i16 3
}

define i16 @vf2i16(ptr %this) readnone {
  ret i16 4
}

define i16 @vf3i16(ptr %this) readnone {
  ret i16 5
}

define i16 @vf4i16(ptr %this) readnone {
  ret i16 6
}

; CHECK: define i1 @call1(
define i1 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: [[VTGEP1:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -1
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
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i16 0, i16 1
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: [[VTGEP2:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -1
  ; CHECK: [[VTLOAD2:%[^ ]*]] = load i8, ptr [[VTGEP2]]
  ; CHECK: [[VTAND2:%[^ ]*]] = and i8 [[VTLOAD2]], 2
  ; CHECK: [[VTCMP2:%[^ ]*]] = icmp ne i8 [[VTAND2]], 0
  %result = call i1 %fptr(ptr %obj)
  ; CHECK: ret i1 [[VTCMP2]]
  ret i1 %result
}

; CHECK: define i16 @call3(
define i16 @call3(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i16 0, i16 2
  %fptr = load ptr, ptr %fptrptr
  ; CHECK: [[VTGEP3:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -4
  ; CHECK: [[VTLOAD3:%[^ ]*]] = load i16, ptr [[VTGEP3]]
  %result = call i16 %fptr(ptr %obj)
  ; CHECK: ret i16 [[VTLOAD3]]
  ret i16 %result
}

; CHECK: define i1 @call1_rel(
define i1 @call1_rel(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid3")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  %result = call i1 %fptr(ptr %obj)
  ret i1 %result
  ; CHECK: [[RES:%[^ ]*]] = icmp eq ptr %vtable, @vt7_rel
  ; CHECK: ret i1 [[RES]]
}

; CHECK: define i1 @call2_rel(
define i1 @call2_rel(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid3")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 4)
  %result = call i1 %fptr(ptr %obj)
  ret i1 %result
  ; CHECK: [[RES:%[^ ]*]] = icmp eq ptr %vtable, @vt6_rel
  ; CHECK: ret i1 [[RES]]
}

; CHECK: define i16 @call3_rel(
define i16 @call3_rel(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid3")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 8)
  ; CHECK: [[VTGEP3:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -2
  ; CHECK: [[VTLOAD3:%[^ ]*]] = load i16, ptr [[VTGEP3]]
  %result = call i16 %fptr(ptr %obj)
  ; CHECK: ret i16 [[VTLOAD3]]
  ret i16 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
declare void @__cxa_pure_virtual()
declare ptr @llvm.load.relative.i32(ptr, i32)

; CHECK: [[T8]] = !{i32 8, !"typeid"}
; CHECK: [[T5]] = !{i32 4, !"typeid"}
; CHECK: [[T16]] = !{i32 16, !"typeid"}
; CHECK: [[T0]] = !{i32 0, !"typeid"}
; CHECK: [[TREL]] = !{i32 4, !"typeid3"}

!0 = !{i32 0, !"typeid"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
