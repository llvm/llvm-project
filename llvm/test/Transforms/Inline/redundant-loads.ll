; RUN: opt -inline < %s -S -o - -inline-threshold=3  | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @pad() readnone

define void @outer1(i32* %a) {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call void @inner1
  %b = alloca i32
  call void @inner1(i32* %a, i32* %b)
  ret void
}

define void @inner1(i32* %a, i32* %b) {
  %1 = load i32, i32* %a
  store i32 %1, i32 * %b ; This store does not clobber the first load.
  %2 = load i32, i32* %a
  call void @pad()
  %3 = load i32, i32* %a
  ret void
}


define void @outer2(i32* %a, i32* %b) {
; CHECK-LABEL: @outer2(
; CHECK: call void @inner2
  call void @inner2(i32* %a, i32* %b)
  ret void
}

define void @inner2(i32* %a, i32* %b) {
  %1 = load i32, i32* %a
  store i32 %1, i32 * %b ; This store clobbers the first load.
  %2 = load i32, i32* %a
  call void @pad()
  ret void
}


define void @outer3(i32* %a) {
; CHECK-LABEL: @outer3(
; CHECK: call void @inner3
  call void @inner3(i32* %a)
  ret void
}

declare void @ext()

define void @inner3(i32* %a) {
  %1 = load i32, i32* %a
  call void @ext() ; This call clobbers the first load.
  %2 = load i32, i32* %a
  ret void
}


define void @outer4(i32* %a, i32* %b, i32* %c) {
; CHECK-LABEL: @outer4(
; CHECK-NOT: call void @inner4
  call void @inner4(i32* %a, i32* %b, i1 false)
  ret void
}

define void @inner4(i32* %a, i32* %b, i1 %pred) {
  %1 = load i32, i32* %a
  br i1 %pred, label %cond_true, label %cond_false

cond_true:
  store i32 %1, i32 * %b ; This store does not clobber the first load.
  br label %cond_false

cond_false:
  %2 = load i32, i32* %a
  call void @pad()
  %3 = load i32, i32* %a
  %4 = load i32, i32* %a
  ret void
}


define void @outer5(i32* %a, double %b) {
; CHECK-LABEL: @outer5(
; CHECK-NOT: call void @inner5
  call void @inner5(i32* %a, double %b)
  ret void
}

declare double @llvm.fabs.f64(double) nounwind readnone

define void @inner5(i32* %a, double %b) {
  %1 = load i32, i32* %a
  %2 = call double @llvm.fabs.f64(double %b) ; This intrinsic does not clobber the first load.
  %3 = load i32, i32* %a
  call void @pad()
  ret void
}

define void @outer6(i32* %a, i8* %ptr) {
; CHECK-LABEL: @outer6(
; CHECK-NOT: call void @inner6
  call void @inner6(i32* %a, i8* %ptr)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) argmemonly nounwind

define void @inner6(i32* %a, i8* %ptr) {
  %1 = load i32, i32* %a
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %ptr) ; This intrinsic does not clobber the first load.
  %2 = load i32, i32* %a
  call void @pad()
  %3 = load i32, i32* %a
  ret void
}

define void @outer7(i32* %a) {
; CHECK-LABEL: @outer7(
; CHECK-NOT: call void @inner7
  call void @inner7(i32* %a)
  ret void
}

declare void @ext2() readnone

define void @inner7(i32* %a) {
  %1 = load i32, i32* %a
  call void @ext2() ; This call does not clobber the first load.
  %2 = load i32, i32* %a
  ret void
}


define void @outer8(i32* %a) {
; CHECK-LABEL: @outer8(
; CHECK-NOT: call void @inner8
  call void @inner8(i32* %a, void ()* @ext2)
  ret void
}

define void @inner8(i32* %a, void ()* %f) {
  %1 = load i32, i32* %a
  call void %f() ; This indirect call does not clobber the first load.
  %2 = load i32, i32* %a
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


define void @outer9(i32* %a) {
; CHECK-LABEL: @outer9(
; CHECK: call void @inner9
  call void @inner9(i32* %a, void ()* @ext)
  ret void
}

define void @inner9(i32* %a, void ()* %f) {
  %1 = load i32, i32* %a
  call void %f() ; This indirect call clobbers the first load.
  %2 = load i32, i32* %a
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
