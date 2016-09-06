; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-block-placement | FileCheck %s

; Test irreducible CFG handling.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; A simple loop with two entries.

; CHECK-LABEL: test0:
; CHECK: f64.load
; CHECK: i32.const $[[REG:[^,]+]]=, 0{{$}}
; CHECK: br_table  $[[REG]],
define void @test0(double* %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
bb:
  %tmp = icmp eq i32 %arg2, 0
  br i1 %tmp, label %bb6, label %bb3

bb3:
  %tmp4 = getelementptr double, double* %arg, i32 %arg3
  %tmp5 = load double, double* %tmp4, align 4
  br label %bb13

bb6:
  %tmp7 = phi i32 [ %tmp18, %bb13 ], [ 0, %bb ]
  %tmp8 = icmp slt i32 %tmp7, %arg1
  br i1 %tmp8, label %bb9, label %bb19

bb9:
  %tmp10 = getelementptr double, double* %arg, i32 %tmp7
  %tmp11 = load double, double* %tmp10, align 4
  %tmp12 = fmul double %tmp11, 2.300000e+00
  store double %tmp12, double* %tmp10, align 4
  br label %bb13

bb13:
  %tmp14 = phi double [ %tmp5, %bb3 ], [ %tmp12, %bb9 ]
  %tmp15 = phi i32 [ undef, %bb3 ], [ %tmp7, %bb9 ]
  %tmp16 = getelementptr double, double* %arg, i32 %tmp15
  %tmp17 = fadd double %tmp14, 1.300000e+00
  store double %tmp17, double* %tmp16, align 4
  %tmp18 = add nsw i32 %tmp15, 1
  br label %bb6

bb19:
  ret void
}

; A simple loop with two entries and an inner natural loop.

; CHECK-LABEL: test1:
; CHECK: f64.load
; CHECK: i32.const $[[REG:[^,]+]]=, 0{{$}}
; CHECK: br_table  $[[REG]],
define void @test1(double* %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
bb:
  %tmp = icmp eq i32 %arg2, 0
  br i1 %tmp, label %bb6, label %bb3

bb3:
  %tmp4 = getelementptr double, double* %arg, i32 %arg3
  %tmp5 = load double, double* %tmp4, align 4
  br label %bb13

bb6:
  %tmp7 = phi i32 [ %tmp18, %bb13 ], [ 0, %bb ]
  %tmp8 = icmp slt i32 %tmp7, %arg1
  br i1 %tmp8, label %bb9, label %bb19

bb9:
  %tmp10 = getelementptr double, double* %arg, i32 %tmp7
  %tmp11 = load double, double* %tmp10, align 4
  %tmp12 = fmul double %tmp11, 2.300000e+00
  store double %tmp12, double* %tmp10, align 4
  br label %bb10

bb10:
  %p = phi i32 [ 0, %bb9 ], [ %pn, %bb10 ]
  %pn = add i32 %p, 1
  %c = icmp slt i32 %pn, 256
  br i1 %c, label %bb10, label %bb13

bb13:
  %tmp14 = phi double [ %tmp5, %bb3 ], [ %tmp12, %bb10 ]
  %tmp15 = phi i32 [ undef, %bb3 ], [ %tmp7, %bb10 ]
  %tmp16 = getelementptr double, double* %arg, i32 %tmp15
  %tmp17 = fadd double %tmp14, 1.300000e+00
  store double %tmp17, double* %tmp16, align 4
  %tmp18 = add nsw i32 %tmp15, 1
  br label %bb6

bb19:
  ret void
}
