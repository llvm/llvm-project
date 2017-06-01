; RUN: opt -instcombine -S < %s | FileCheck %s
target datalayout = "E-p:64:64:64-p1:64:64:64-p2:32:32:32-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@x = external global <2 x i64>, align 16
@xx = external global [13 x <2 x i64>], align 16

@x.as2 = external addrspace(2) global <2 x i64>, align 16

; CHECK-LABEL: @static_hem(
; CHECK: , align 16
define <2 x i64> @static_hem() {
  %t = getelementptr <2 x i64>, <2 x i64>* @x, i32 7
  %tmp1 = load <2 x i64>, <2 x i64>* %t, align 1
  ret <2 x i64> %tmp1
}

; CHECK-LABEL: @hem(
; CHECK: , align 16
define <2 x i64> @hem(i32 %i) {
  %t = getelementptr <2 x i64>, <2 x i64>* @x, i32 %i
  %tmp1 = load <2 x i64>, <2 x i64>* %t, align 1
  ret <2 x i64> %tmp1
}

; CHECK-LABEL: @hem_2d(
; CHECK: , align 16
define <2 x i64> @hem_2d(i32 %i, i32 %j) {
  %t = getelementptr [13 x <2 x i64>], [13 x <2 x i64>]* @xx, i32 %i, i32 %j
  %tmp1 = load <2 x i64>, <2 x i64>* %t, align 1
  ret <2 x i64> %tmp1
}

; CHECK-LABEL: @foo(
; CHECK: , align 16
define <2 x i64> @foo() {
  %tmp1 = load <2 x i64>, <2 x i64>* @x, align 1
  ret <2 x i64> %tmp1
}

; CHECK-LABEL: @bar(
; CHECK: , align 16
; CHECK: , align 16
define <2 x i64> @bar() {
  %t = alloca <2 x i64>
  call void @kip(<2 x i64>* %t)
  %tmp1 = load <2 x i64>, <2 x i64>* %t, align 1
  ret <2 x i64> %tmp1
}

; CHECK-LABEL: @static_hem_store(
; CHECK: , align 16
define void @static_hem_store(<2 x i64> %y) {
  %t = getelementptr <2 x i64>, <2 x i64>* @x, i32 7
  store <2 x i64> %y, <2 x i64>* %t, align 1
  ret void
}

; CHECK-LABEL: @hem_store(
; CHECK: , align 16
define void @hem_store(i32 %i, <2 x i64> %y) {
  %t = getelementptr <2 x i64>, <2 x i64>* @x, i32 %i
  store <2 x i64> %y, <2 x i64>* %t, align 1
  ret void
}

; CHECK-LABEL: @hem_2d_store(
; CHECK: , align 16
define void @hem_2d_store(i32 %i, i32 %j, <2 x i64> %y) {
  %t = getelementptr [13 x <2 x i64>], [13 x <2 x i64>]* @xx, i32 %i, i32 %j
  store <2 x i64> %y, <2 x i64>* %t, align 1
  ret void
}

; CHECK-LABEL: @foo_store(
; CHECK: , align 16
define void @foo_store(<2 x i64> %y) {
  store <2 x i64> %y, <2 x i64>* @x, align 1
  ret void
}

; CHECK-LABEL: @bar_store(
; CHECK: , align 16
define void @bar_store(<2 x i64> %y) {
  %t = alloca <2 x i64>
  call void @kip(<2 x i64>* %t)
  store <2 x i64> %y, <2 x i64>* %t, align 1
  ret void
}

declare void @kip(<2 x i64>* %t)
