; RUN: opt < %s -basic-aa -dse -enable-dse-memoryssa -S | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=dse -enable-dse-memoryssa -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
declare void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* nocapture, i8, i64, i32) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind
declare void @llvm.init.trampoline(i8*, i8*, i8*)

; **** Noop load->store tests **************************************************

; We CAN optimize volatile loads.
define void @test_load_volatile(i32* %Q) {
; CHECK-LABEL: @test_load_volatile(
; CHECK-NEXT:    [[A:%.*]] = load volatile i32, i32* [[Q:%.*]]
; CHECK-NEXT:    store i32 [[A]], i32* [[Q]]
; CHECK-NEXT:    ret void
;
  %a = load volatile i32, i32* %Q
  store i32 %a, i32* %Q
  ret void
}

; We can NOT optimize volatile stores.
define void @test_store_volatile(i32* %Q) {
; CHECK-LABEL: @test_store_volatile(
; CHECK-NEXT:    [[A:%.*]] = load i32, i32* [[Q:%.*]]
; CHECK-NEXT:    store volatile i32 [[A]]
; CHECK-NEXT:    ret void
;
  %a = load i32, i32* %Q
  store volatile i32 %a, i32* %Q
  ret void
}

; PR2599 - load -> store to same address.
define void @test12({ i32, i32 }* %x) nounwind  {
; CHECK-LABEL: @test12(
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr { i32, i32 }, { i32, i32 }* [[X:%.*]], i32 0, i32 1
; CHECK-NEXT:    [[TMP8:%.*]] = load i32, i32* [[TMP7]], align 4
; CHECK-NEXT:    [[TMP17:%.*]] = sub i32 0, [[TMP8]]
; CHECK-NEXT:    store i32 [[TMP17]], i32* [[TMP7]], align 4
; CHECK-NEXT:    ret void
;
  %tmp4 = getelementptr { i32, i32 }, { i32, i32 }* %x, i32 0, i32 0
  %tmp5 = load i32, i32* %tmp4, align 4
  %tmp7 = getelementptr { i32, i32 }, { i32, i32 }* %x, i32 0, i32 1
  %tmp8 = load i32, i32* %tmp7, align 4
  %tmp17 = sub i32 0, %tmp8
  store i32 %tmp5, i32* %tmp4, align 4
  store i32 %tmp17, i32* %tmp7, align 4
  ret void
}

; Remove redundant store if loaded value is in another block.
define i32 @test26(i1 %c, i32* %p) {
; CHECK-LABEL: @test26(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    ret i32 0
;
entry:
  %v = load i32, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  store i32 %v, i32* %p, align 4
  br label %bb3
bb3:
  ret i32 0
}

; Remove redundant store if loaded value is in another block.
define i32 @test27(i1 %c, i32* %p) {
; CHECK-LABEL: @test27(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    ret i32 0
;
entry:
  %v = load i32, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  br label %bb3
bb3:
  store i32 %v, i32* %p, align 4
  ret i32 0
}

; Remove redundant store if loaded value is in another block inside a loop.
define i32 @test31(i1 %c, i32* %p, i32 %i) {
; CHECK-LABEL: @test31(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1]], label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    ret i32 0
;
entry:
  %v = load i32, i32* %p, align 4
  br label %bb1
bb1:
  store i32 %v, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb2:
  ret i32 0
}

; Don't remove "redundant" store if %p is possibly stored to.
define i32 @test46(i1 %c, i32* %p, i32* %p2, i32 %i) {
; CHECK-LABEL: @test46(
; CHECK:  load
; CHECK:  store
; CHECK:  store
; CHECK:  ret i32 0
;
entry:
  %v = load i32, i32* %p, align 4
  br label %bb1
bb1:
  store i32 %v, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb2:
  store i32 0, i32* %p2, align 4
  br i1 %c, label %bb3, label %bb1
bb3:
  ret i32 0
}

declare void @unknown_func()

; Remove redundant store, which is in the lame loop as the load.
define i32 @test33(i1 %c, i32* %p, i32 %i) {
; CHECK-LABEL: @test33(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    call void @unknown_func()
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1]], label [[BB3:%.*]]
; CHECK:       bb3:
; CHECK-NEXT:    ret i32 0
;
entry:
  br label %bb1
bb1:
  %v = load i32, i32* %p, align 4
  br label %bb2
bb2:
  store i32 %v, i32* %p, align 4
  ; Might read and overwrite value at %p, but doesn't matter.
  call void @unknown_func()
  br i1 %c, label %bb1, label %bb3
bb3:
  ret i32 0
}

declare void @unkown_write(i32*)

; We can't remove the "noop" store around an unkown write.
define void @test43(i32* %Q) {
; CHECK-LABEL: @test43(
; CHECK-NEXT:    load
; CHECK-NEXT:    call
; CHECK-NEXT:    store
; CHECK-NEXT:    ret void
;
  %a = load i32, i32* %Q
  call void @unkown_write(i32* %Q)
  store i32 %a, i32* %Q
  ret void
}

; We CAN remove it when the unkown write comes AFTER.
define void @test44(i32* %Q) {
; CHECK-LABEL: @test44(
; CHECK-NEXT:    call
; CHECK-NEXT:    ret void
  %a = load i32, i32* %Q
  store i32 %a, i32* %Q
  call void @unkown_write(i32* %Q)
  ret void
}

define void @test45(i32* %Q) {
; CHECK-LABEL: @test45(
; CHECK-NEXT:    [[A:%.*]] = load
; CHECK-NEXT:    store i32 [[A]]
; CHECK-NEXT:    ret void
  %a = load i32, i32* %Q
  store i32 10, i32* %Q
  store i32 %a, i32* %Q
  ret void
}

define i32 @test48(i1 %c, i32* %p) {
; CHECK-LABEL: @test48(
; CHECK: entry:
; CHECK-NEXT: [[V:%.*]] = load
; CHECK: store i32 0
; CHECK: store i32 [[V]]
; CHECK: ret i32 0
entry:
  %v = load i32, i32* %p, align 4
  br i1 %c, label %bb0, label %bb0.0

bb0:
  store i32 0, i32* %p
  br i1 %c, label %bb1, label %bb2

bb0.0:
  br label %bb1

bb1:
  store i32 %v, i32* %p, align 4
  br i1 %c, label %bb2, label %bb0
bb2:
  ret i32 0
}

; TODO: Remove both redundant stores if loaded value is in another block inside a loop.
define i32 @test47(i1 %c, i32* %p, i32 %i) {
; X-CHECK-LABEL: @test47(
; X-CHECK-NEXT:  entry:
; X-CHECK-NEXT:    br label [[BB1:%.*]]
; X-CHECK:       bb1:
; X-CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1]], label [[BB2:%.*]]
; X-CHECK:       bb2:
; X-CHECK-NEXT:    br i1 [[C]], label [[BB2]], label [[BB3:%.*]]
; X-CHECK:       bb3:
; X-CHECK-NEXT:    ret i32 0
entry:
  %v = load i32, i32* %p, align 4
  br label %bb1
bb1:
  store i32 %v, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb2:
  store i32 %v, i32* %p, align 4
  br i1 %c, label %bb3, label %bb1
bb3:
  ret i32 0
}

