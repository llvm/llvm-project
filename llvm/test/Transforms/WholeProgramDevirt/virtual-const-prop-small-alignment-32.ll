;; This target uses 32-bit sized and aligned pointers.
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility --data-layout="e-p:32:32-i64:64:64" %s | FileCheck %s

;; The tests should be the exact same even with different preferred alignments since
;; the ABI alignment is used.
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility --data-layout="e-p:32:32-i64:64:128" %s | FileCheck %s

;; Constant propagation should be agnostic towards sections.
;; Also the new global should be in the original vtable's section.
; CHECK:      [[VT1DATA:@[^ ]*]] = {{.*}} { [4 x i8], [3 x ptr], [0 x i8] }
; CHECK-SAME:   [4 x i8]  c"\00\00\03\00",
; CHECK-SAME: }, section "vt1sec", !type [[T8:![0-9]+]]
@vt1 = constant [3 x ptr] [
ptr @vf0i1,
ptr @vf1i8,
ptr @vf1i32
], section "vt1sec", !type !0

;; This represents a normal vtable using the default ABI alignments.
;; For this test, the pointers are 32-bit aligned. The important things
;; to note are:
;;
;; 1. All the constants from the 3rd field propgating an i32 placed at an
;;    offset of -6 from the vtable. Since this vtable is 32-bit aligned
;;    according to the datalayout, this could result in an unaligned load.
;; 2. The call instruction in @call3 is replaced with a GEP + load.
;;
; CHECK:      [[VT2DATA:@[^ ]*]] = {{.*}} { [4 x i8], [3 x ptr], [0 x i8] }
; CHECK-SAME:   [4 x i8] c"\00\00\02\01",
; CHECK-SAME: !type [[T8]]
@vt2 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i8,
ptr @vf2i32
], !type !0

;; This represents an underaligned vtable.
;;
;; All the functions returning i8s and i1s should still be constant-propagated
;; because we can still do an aligned load regardless of where the 1-byte aligned
;; vtable is.
; CHECK:      [[VT3DATA:@[^ ]*]] = {{.*}} { [2 x i8], [3 x ptr], [0 x i8] }
; CHECK-SAME:   [2 x i8] c"\03\00",
; CHECK-SAME: }, align 1, !type [[T5:![0-9]+]]
@vt3 = constant [3 x ptr] [
ptr @vf0i1,
ptr @vf1i8,
ptr @vf3i32
], align 1, !type !0

;; This represents an overaligned vtable.
; CHECK:      [[VT4DATA:@[^ ]*]] = {{.*}} { [16 x i8], [3 x ptr], [0 x i8] }
; CHECK-SAME:   [16 x i8] c"\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\01",
; CHECK-SAME: },  align 16, !type [[T16:![0-9]+]]
@vt4 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf0i8,
ptr @vf4i32
], align 16, !type !0

;; These contain a mix of different integral type sizes.
; CHECK:      [[VT6DATA:@[^ ]*]] = {{.*}} { [4 x i8], [3 x ptr], [0 x i8] } 
; CHECK-SAME:   [4 x i8] c"\00\00\00\0B",
; CHECK-SAME: }, !type [[T1:![0-9]+]]
@vt6 = constant [3 x ptr] [
ptr @vf0i1,
ptr @vf10i8,
ptr @vf5i64
], !type !1

; CHECK:      [[VT7DATA:@[^ ]*]] = {{.*}} { [4 x i8], [3 x ptr], [0 x i8] }
; CHECK-SAME:   [4 x i8] c"\00\00\00\0A",
; CHECK-SAME: }, !type [[T1]]
@vt7 = constant [3 x ptr] [
ptr @vf1i1,
ptr @vf9i8,
ptr @vf6i64
], !type !1

;; Test relative vtables
; CHECK:      [[VT6RELDATA:@[^ ]*]] = {{.*}} { [4 x i8], [3 x i32], [0 x i8] }
; CHECK-SAME:     [4 x i8] c"\00\00\00\0B",
; CHECK-SAME: ], [0 x i8] zeroinitializer }, !type [[TREL:![0-9]+]]
@vt6_rel = constant [3 x i32] [
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf0i1 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf10i8 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf5i64 to i64), i64 ptrtoint (ptr @vt6_rel to i64)) to i32)
], !type !2

; CHECK:      [[VT7RELDATA:@[^ ]*]] = {{.*}} { [4 x i8], [3 x i32], [0 x i8] } 
; CHECK-SAME:   [4 x i8] c"\00\00\00\0A",
; CHECK-SAME: ], [0 x i8] zeroinitializer }, !type [[TREL]]
@vt7_rel = constant [3 x i32] [
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1i1 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf9i8 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf6i64 to i64), i64 ptrtoint (ptr @vt7_rel to i64)) to i32)
], !type !2

; CHECK: @vt1 = alias [3 x ptr], getelementptr inbounds ({ [4 x i8], [3 x ptr], [0 x i8] }, ptr [[VT1DATA]], i32 0, i32 1)
; CHECK: @vt2 = alias [3 x ptr], getelementptr inbounds ({ [4 x i8], [3 x ptr], [0 x i8] }, ptr [[VT2DATA]], i32 0, i32 1)
; CHECK: @vt3 = alias [3 x ptr], getelementptr inbounds ({ [2 x i8], [3 x ptr], [0 x i8] }, ptr [[VT3DATA]], i32 0, i32 1)
; CHECK: @vt4 = alias [3 x ptr], getelementptr inbounds ({ [16 x i8], [3 x ptr], [0 x i8] }, ptr [[VT4DATA]], i32 0, i32 1)
; CHECK: @vt6 = alias [3 x ptr], getelementptr inbounds ({ [4 x i8], [3 x ptr], [0 x i8] }, ptr [[VT6DATA]], i32 0, i32 1)
; CHECK: @vt7 = alias [3 x ptr], getelementptr inbounds ({ [4 x i8], [3 x ptr], [0 x i8] }, ptr [[VT7DATA]], i32 0, i32 1)
; CHECK: @vt6_rel = alias [3 x i32], getelementptr inbounds ({ [4 x i8], [3 x i32], [0 x i8] }, ptr [[VT6RELDATA]], i32 0, i32 1)
; CHECK: @vt7_rel = alias [3 x i32], getelementptr inbounds ({ [4 x i8], [3 x i32], [0 x i8] }, ptr [[VT7RELDATA]], i32 0, i32 1)

define i1 @vf0i1(ptr %this) readnone {
  ret i1 0
}

define i1 @vf1i1(ptr %this) readnone {
  ret i1 1
}

define i8 @vf0i8(ptr %this) readnone {
  ret i8 2
}

define i8 @vf1i8(ptr %this) readnone {
  ret i8 3
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

define i64 @vf5i64(ptr %this) readnone {
  ret i64 5
}

define i64 @vf6i64(ptr %this) readnone {
  ret i64 6
}

define i16 @vf7i16(ptr %this) readnone {
  ret i16 7
}

define i16 @vf8i16(ptr %this) readnone {
  ret i16 8
}

define i8 @vf9i8(ptr %this) readnone {
  ret i8 10
}

define i8 @vf10i8(ptr %this) readnone {
  ret i8 11
}

; CHECK-LABEL: define i1 @call1(
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

; CHECK-LABEL: define i8 @call2(
define i8 @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 1
  %fptr = load ptr, ptr %fptrptr
  %result = call i8 %fptr(ptr %obj)
  ret i8 %result
  ; CHECK: [[VTGEP2:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -2
  ; CHECK: [[VTLOAD:%[^ ]*]] = load i8, ptr [[VTGEP2]]
  ; CHECK: ret i8 [[VTLOAD]]
}

;; We never constant propagate this since the i32 cannot be reliably loaded
;; without misalignment from all "typeid" vtables (due to the `align 1` vtable).
; CHECK-LABEL: define i32 @call3(
define i32 @call3(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 2
  %fptr = load ptr, ptr %fptrptr
  %result = call i32 %fptr(ptr %obj)
  ret i32 %result
  ; CHECK: [[FPTRPTR:%.*]] = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 2
  ; CHECK: [[FPTR:%.*]] = load ptr, ptr [[FPTRPTR]], align 4
  ; CHECK: [[RES:%.*]] = call i32 [[FPTR]](ptr %obj)
  ; CHECK: ret i32 [[RES]]
}

; CHECK-LABEL: define i1 @call4(
define i1 @call4(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 0
  %fptr = load ptr, ptr %fptrptr
  %result = call i1 %fptr(ptr %obj)
  ret i1 %result
  ; CHECK: [[RES:%[^ ]*]] = icmp eq ptr %vtable, @vt7
  ; CHECK: ret i1 [[RES]]
}

; CHECK-LABEL: define i64 @call5(
define i64 @call5(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 2
  %fptr = load ptr, ptr %fptrptr
  %result = call i64 %fptr(ptr %obj)
  ret i64 %result
  ; CHECK: [[FPTRPTR:%.*]] = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 2
  ; CHECK: [[FPTR:%.*]] = load ptr, ptr [[FPTRPTR]], align 4
  ; CHECK: [[RES:%.*]] = call i64 [[FPTR]](ptr %obj)
  ; CHECK: ret i64 [[RES]]
}

; CHECK-LABEL: define i8 @call6(
define i8 @call6(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 1
  %fptr = load ptr, ptr %fptrptr
  %result = call i8 %fptr(ptr %obj)
  ret i8 %result
  ; CHECK: [[VTGEP2:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -1
  ; CHECK: [[VTLOAD:%[^ ]*]] = load i8, ptr [[VTGEP2]]
  ; CHECK: ret i8 [[VTLOAD]]
}

; CHECK-LABEL: define i1 @call4_rel(
define i1 @call4_rel(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid3")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  %result = call i1 %fptr(ptr %obj)
  ret i1 %result
  ; CHECK: [[RES:%[^ ]*]] = icmp eq ptr %vtable, @vt7_rel
  ; CHECK: ret i1 [[RES]]
}

; CHECK-LABEL: define i64 @call5_rel(
define i64 @call5_rel(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid3")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 8)
  %result = call i64 %fptr(ptr %obj)
  ret i64 %result
  ; CHECK: [[FPTR:%.*]] = call ptr @llvm.load.relative.i32(ptr %vtable, i32 8)
  ; CHECK: [[RES:%.*]] = call i64 [[FPTR]](ptr %obj)
  ; CHECK: ret i64 [[RES]]
}

; CHECK-LABEL: define i8 @call6_rel(
define i8 @call6_rel(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid3")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 4)
  %result = call i8 %fptr(ptr %obj)
  ret i8 %result
  ; CHECK: [[VTGEP2:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -1
  ; CHECK: [[VTLOAD:%[^ ]*]] = load i8, ptr [[VTGEP2]]
  ; CHECK: ret i8 [[VTLOAD]]
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
declare void @__cxa_pure_virtual()
declare ptr @llvm.load.relative.i32(ptr, i32)

; CHECK: [[T8]] = !{i32 4, !"typeid"}
; CHECK: [[T5]] = !{i32 2, !"typeid"}
; CHECK: [[T16]] = !{i32 16, !"typeid"}
; CHECK: [[T1]] = !{i32 4, !"typeid2"}
; CHECK: [[TREL]] = !{i32 4, !"typeid3"}

!0 = !{i32 0, !"typeid"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
