; RUN: opt -loop-reduce %s -S -o - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32"
target triple = "i686-pc-win32"

; <rdar://problem/14199725> Assertion failed: (CurScaleCost >= 0 && "Legal addressing mode has an illegal cost!")
; CHECK-LABEL: @scalingFactorCrash(
define void @scalingFactorCrash(i1 %arg) {
  br i1 %arg, label %1, label %24

; <label>:1                                       ; preds = %0
  br i1 %arg, label %2, label %24

; <label>:2                                       ; preds = %1
  br i1 %arg, label %3, label %24

; <label>:3                                       ; preds = %2
  br i1 %arg, label %4, label %24

; <label>:4                                       ; preds = %3
  br i1 %arg, label %24, label %6

; <label>:5                                       ; preds = %6
  br i1 %arg, label %24, label %7

; <label>:6                                       ; preds = %6, %4
  br i1 %arg, label %6, label %5

; <label>:7                                       ; preds = %9, %5
  br label %8

; <label>:8                                       ; preds = %8, %7
  br i1 %arg, label %9, label %8

; <label>:9                                       ; preds = %8
  br i1 %arg, label %7, label %10

; <label>:10                                      ; preds = %9
  br i1 %arg, label %24, label %11

; <label>:11                                      ; preds = %10
  br i1 %arg, label %15, label %13

; <label>:12                                      ; preds = %14
  br label %15

; <label>:13                                      ; preds = %11
  br label %14

; <label>:14                                      ; preds = %14, %13
  br i1 %arg, label %14, label %12

; <label>:15                                      ; preds = %12, %11
  br i1 %arg, label %16, label %24

; <label>:16                                      ; preds = %16, %15
  %17 = phi i32 [ %21, %16 ], [ undef, %15 ]
  %18 = sub i32 %17, 1623127498
  %19 = getelementptr inbounds i32, ptr undef, i32 %18
  store i32 undef, ptr %19, align 4
  %20 = add i32 %17, 1623127499
  %21 = add i32 %20, -1623127498
  %22 = add i32 %21, -542963121
  %23 = icmp ult i32 %22, undef
  br i1 %arg, label %16, label %24

; <label>:24                                      ; preds = %16, %15, %10, %5, %4, %3, %2, %1, %0
  ret void
}
