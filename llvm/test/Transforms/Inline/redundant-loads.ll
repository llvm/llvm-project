; RUN: opt -passes=inline < %s -S -o - -inline-threshold=3  | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @pad() readnone

define void @outer1(ptr %a) {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call void @inner1
  %b = alloca i32
  call void @inner1(ptr %a, ptr %b)
  ret void
}

define void @inner1(ptr %a, ptr %b) {
  %1 = load i32, ptr %a
  store i32 %1, ptr %b ; This store does not clobber the first load.
  %2 = load i32, ptr %a
  call void @pad()
  %3 = load i32, ptr %a
  ret void
}


define void @outer2(ptr %a, ptr %b) {
; CHECK-LABEL: @outer2(
; CHECK: call void @inner2
  call void @inner2(ptr %a, ptr %b)
  ret void
}

define void @inner2(ptr %a, ptr %b) {
  %1 = load i32, ptr %a
  store i32 %1, ptr %b ; This store clobbers the first load.
  %2 = load i32, ptr %a
  call void @pad()
  ret void
}


define void @outer3(ptr %a) {
; CHECK-LABEL: @outer3(
; CHECK: call void @inner3
  call void @inner3(ptr %a)
  ret void
}

declare void @ext()

define void @inner3(ptr %a) {
  %1 = load i32, ptr %a
  call void @ext() ; This call clobbers the first load.
  %2 = load i32, ptr %a
  ret void
}


define void @outer4(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: @outer4(
; CHECK-NOT: call void @inner4
  call void @inner4(ptr %a, ptr %b, i1 false)
  ret void
}

define void @inner4(ptr %a, ptr %b, i1 %pred) {
  %1 = load i32, ptr %a
  br i1 %pred, label %cond_true, label %cond_false

cond_true:
  store i32 %1, ptr %b ; This store does not clobber the first load.
  br label %cond_false

cond_false:
  %2 = load i32, ptr %a
  call void @pad()
  %3 = load i32, ptr %a
  %4 = load i32, ptr %a
  ret void
}


define void @outer5(ptr %a, double %b) {
; CHECK-LABEL: @outer5(
; CHECK-NOT: call void @inner5
  call void @inner5(ptr %a, double %b)
  ret void
}

declare double @llvm.fabs.f64(double) nounwind readnone

define void @inner5(ptr %a, double %b) {
  %1 = load i32, ptr %a
  %2 = call double @llvm.fabs.f64(double %b) ; This intrinsic does not clobber the first load.
  %3 = load i32, ptr %a
  call void @pad()
  ret void
}

define void @outer6(ptr %a, ptr %ptr) {
; CHECK-LABEL: @outer6(
; CHECK-NOT: call void @inner6
  call void @inner6(ptr %a, ptr %ptr)
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) argmemonly nounwind

define void @inner6(ptr %a, ptr %ptr) {
  %1 = load i32, ptr %a
  call void @llvm.lifetime.start.p0(i64 32, ptr %ptr) ; This intrinsic does not clobber the first load.
  %2 = load i32, ptr %a
  call void @pad()
  %3 = load i32, ptr %a
  ret void
}

define void @outer7(ptr %a) {
; CHECK-LABEL: @outer7(
; CHECK-NOT: call void @inner7
  call void @inner7(ptr %a)
  ret void
}

declare void @ext2() readnone

define void @inner7(ptr %a) {
  %1 = load i32, ptr %a
  call void @ext2() ; This call does not clobber the first load.
  %2 = load i32, ptr %a
  ret void
}


define void @outer8(ptr %a) {
; CHECK-LABEL: @outer8(
; CHECK-NOT: call void @inner8
  call void @inner8(ptr %a, ptr @ext2)
  ret void
}

define void @inner8(ptr %a, ptr %f) {
  %1 = load i32, ptr %a
  call void %f() ; This indirect call does not clobber the first load.
  %2 = load i32, ptr %a
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  ret void
}


define void @outer9(ptr %a) {
; CHECK-LABEL: @outer9(
; CHECK: call void @inner9
  call void @inner9(ptr %a, ptr @ext)
  ret void
}

define void @inner9(ptr %a, ptr %f) {
  %1 = load i32, ptr %a
  call void %f() ; This indirect call clobbers the first load.
  %2 = load i32, ptr %a
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  call void @pad()
  ret void
}


define void @outer10(ptr %a) {
; CHECK-LABEL: @outer10(
; CHECK: call void @inner10
  %b = alloca i32
  call void @inner10(ptr %a, ptr %b)
  ret void
}

define void @inner10(ptr %a, ptr %b) {
  %1 = load i32, ptr %a
  store i32 %1, ptr %b
  %2 = load volatile i32, ptr %a ; volatile load should be kept.
  call void @pad()
  %3 = load volatile i32, ptr %a ; Same as the above.
  ret void
}
