; RUN: llc -verify-machineinstrs < %s -mcpu=g5 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @foo(ptr noalias nocapture %a, ptr noalias nocapture %b) #0 {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds float, ptr %b, i64 %index
  %wide.load = load <4 x float>, ptr %0, align 4
  %.sum11 = or i64 %index, 4
  %1 = getelementptr float, ptr %b, i64 %.sum11
  %wide.load8 = load <4 x float>, ptr %1, align 4
  %2 = fadd <4 x float> %wide.load, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %3 = fadd <4 x float> %wide.load8, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %4 = getelementptr inbounds float, ptr %a, i64 %index
  store <4 x float> %2, ptr %4, align 4
  %.sum12 = or i64 %index, 4
  %5 = getelementptr float, ptr %a, i64 %.sum12
  store <4 x float> %3, ptr %5, align 4
  %index.next = add i64 %index, 8
  %6 = icmp eq i64 %index.next, 16000
  br i1 %6, label %for.end, label %vector.body

; CHECK: @foo
; CHECK-DAG: li [[C0:[0-9]+]], 0
; CHECK-DAG: lvx [[CNST:[0-9]+]],
; CHECK: .LBB0_1:
; CHECK-DAG: lvsl [[MASK1:[0-9]+]], [[B1:[0-9]+]], [[C0]]
; CHECK-DAG: add [[B3:[0-9]+]], [[B1]], [[C0]]
; CHECK-DAG: lvx [[LD1:[0-9]+]], [[B1]], [[C0]]
; CHECK-DAG: lvx [[LD2:[0-9]+]], [[B3]],
; CHECK-DAG: vperm [[R1:[0-9]+]], [[LD1]], [[LD2]], [[MASK1]]
; CHECK-DAG: vaddfp {{[0-9]+}}, [[R1]], [[CNST]]
; CHECK: blr

for.end:                                          ; preds = %vector.body
  ret void
}

attributes #0 = { nounwind }
