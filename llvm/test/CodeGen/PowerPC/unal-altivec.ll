; RUN: llc -verify-machineinstrs < %s -mcpu=g5 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @foo(float* noalias nocapture %a, float* noalias nocapture %b) #0 {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds float, float* %b, i64 %index
  %1 = bitcast float* %0 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %1, align 4
  %.sum11 = or i64 %index, 4
  %2 = getelementptr float, float* %b, i64 %.sum11
  %3 = bitcast float* %2 to <4 x float>*
  %wide.load8 = load <4 x float>, <4 x float>* %3, align 4
  %4 = fadd <4 x float> %wide.load, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %5 = fadd <4 x float> %wide.load8, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %6 = getelementptr inbounds float, float* %a, i64 %index
  %7 = bitcast float* %6 to <4 x float>*
  store <4 x float> %4, <4 x float>* %7, align 4
  %.sum12 = or i64 %index, 4
  %8 = getelementptr float, float* %a, i64 %.sum12
  %9 = bitcast float* %8 to <4 x float>*
  store <4 x float> %5, <4 x float>* %9, align 4
  %index.next = add i64 %index, 8
  %10 = icmp eq i64 %index.next, 16000
  br i1 %10, label %for.end, label %vector.body

; CHECK: @foo
; CHECK-DAG: li [[C0:[0-9]+]], 0
; CHECK-DAG: li [[C15:[0-9]+]], 15
; CHECK-DAG: lvx [[CNST:[0-9]+]],
; CHECK: .LBB0_1:
; CHECK-DAG: lvsl [[MASK1:[0-9]+]], [[B1:[0-9]+]], [[C0]]
; CHECK-DAG: lvsl [[MASK2:[0-9]+]], [[B2:[0-9]+]], [[C0]]
; CHECK-DAG: add [[B3:[0-9]+]], [[B1]], [[C0]]
; CHECK-DAG: add [[B4:[0-9]+]], [[B2]], [[C0]]
; CHECK-DAG: lvx [[LD1:[0-9]+]], [[B1]], [[C0]]
; CHECK-DAG: lvx [[LD2:[0-9]+]], [[B3]], [[C15]]
; CHECK-DAG: lvx [[LD3:[0-9]+]], [[B2]], [[C0]]
; CHECK-DAG: lvx [[LD4:[0-9]+]], [[B4]], [[C15]]
; CHECK-DAG: vperm [[R1:[0-9]+]], [[LD1]], [[LD2]], [[MASK1]]
; CHECK-DAG: vperm [[R2:[0-9]+]], [[LD3]], [[LD4]], [[MASK2]]
; CHECK-DAG: vaddfp {{[0-9]+}}, [[R1]], [[CNST]]
; CHECK-DAG: vaddfp {{[0-9]+}}, [[R2]], [[CNST]]
; CHECK: blr

for.end:                                          ; preds = %vector.body
  ret void
}

attributes #0 = { nounwind }
