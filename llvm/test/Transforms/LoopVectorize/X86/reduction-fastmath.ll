; RUN: opt -S -loop-vectorize < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @reduction_sum_float_ieee(i32 %n, float* %array) {
; CHECK-LABEL: define float @reduction_sum_float_ieee(
entry:
  %entry.cond = icmp ne i32 0, 4096
  br i1 %entry.cond, label %loop, label %loop.exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %sum = phi float [ 0.000000e+00, %entry ], [ %sum.inc, %loop ]
  %address = getelementptr float, float* %array, i32 %idx
  %value = load float, float* %address
  %sum.inc = fadd float %sum, %value
  %idx.inc = add i32 %idx, 1
  %be.cond = icmp ne i32 %idx.inc, 4096
  br i1 %be.cond, label %loop, label %loop.exit

loop.exit:
  %sum.lcssa = phi float [ %sum.inc, %loop ], [ 0.000000e+00, %entry ]
; CHECK-NOT: %wide.load = load <4 x float>, <4 x float>*
; CHECK: ret float %sum.lcssa
  ret float %sum.lcssa
}

define float @reduction_sum_float_fastmath(i32 %n, float* %array) {
; CHECK-LABEL: define float @reduction_sum_float_fastmath(
; CHECK: fadd fast <4 x float>
; CHECK: fadd fast <4 x float>
; CHECK: fadd fast <4 x float>
; CHECK: fadd fast <4 x float>
; CHECK: fadd fast <4 x float>
entry:
  %entry.cond = icmp ne i32 0, 4096
  br i1 %entry.cond, label %loop, label %loop.exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %sum = phi float [ 0.000000e+00, %entry ], [ %sum.inc, %loop ]
  %address = getelementptr float, float* %array, i32 %idx
  %value = load float, float* %address
  %sum.inc = fadd fast float %sum, %value
  %idx.inc = add i32 %idx, 1
  %be.cond = icmp ne i32 %idx.inc, 4096
  br i1 %be.cond, label %loop, label %loop.exit

loop.exit:
  %sum.lcssa = phi float [ %sum.inc, %loop ], [ 0.000000e+00, %entry ]
; CHECK: ret float %sum.lcssa
  ret float %sum.lcssa
}

define float @reduction_sum_float_only_reassoc(i32 %n, float* %array) {
; CHECK-LABEL: define float @reduction_sum_float_only_reassoc(
; CHECK-NOT: fadd fast
; CHECK: fadd reassoc <4 x float>
; CHECK: fadd reassoc <4 x float>
; CHECK: fadd reassoc <4 x float>
; CHECK: fadd reassoc <4 x float>
; CHECK: fadd reassoc <4 x float>

entry:
  %entry.cond = icmp ne i32 0, 4096
  br i1 %entry.cond, label %loop, label %loop.exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %sum = phi float [ 0.000000e+00, %entry ], [ %sum.inc, %loop ]
  %address = getelementptr float, float* %array, i32 %idx
  %value = load float, float* %address
  %sum.inc = fadd reassoc float %sum, %value
  %idx.inc = add i32 %idx, 1
  %be.cond = icmp ne i32 %idx.inc, 4096
  br i1 %be.cond, label %loop, label %loop.exit

loop.exit:
  %sum.lcssa = phi float [ %sum.inc, %loop ], [ 0.000000e+00, %entry ]
; CHECK: ret float %sum.lcssa
  ret float %sum.lcssa
}

define float @reduction_sum_float_only_reassoc_and_contract(i32 %n, float* %array) {
; CHECK-LABEL: define float @reduction_sum_float_only_reassoc_and_contract(
; CHECK-NOT: fadd fast
; CHECK: fadd reassoc contract <4 x float>
; CHECK: fadd reassoc contract <4 x float>
; CHECK: fadd reassoc contract <4 x float>
; CHECK: fadd reassoc contract <4 x float>
; CHECK: fadd reassoc contract <4 x float>

entry:
  %entry.cond = icmp ne i32 0, 4096
  br i1 %entry.cond, label %loop, label %loop.exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %sum = phi float [ 0.000000e+00, %entry ], [ %sum.inc, %loop ]
  %address = getelementptr float, float* %array, i32 %idx
  %value = load float, float* %address
  %sum.inc = fadd reassoc contract float %sum, %value
  %idx.inc = add i32 %idx, 1
  %be.cond = icmp ne i32 %idx.inc, 4096
  br i1 %be.cond, label %loop, label %loop.exit

loop.exit:
  %sum.lcssa = phi float [ %sum.inc, %loop ], [ 0.000000e+00, %entry ]
; CHECK: ret float %sum.lcssa
  ret float %sum.lcssa
}
