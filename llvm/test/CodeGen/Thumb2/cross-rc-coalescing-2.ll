; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -mcpu=cortex-a8 -arm-atomic-cfg-tidy=0 | FileCheck %s

define void @fht(float* nocapture %fz, i16 signext %n) nounwind {
; CHECK-LABEL: fht:
entry:
  br label %bb5

bb5:                                              ; preds = %bb5, %entry
  br i1 undef, label %bb5, label %bb.nph

bb.nph:                                           ; preds = %bb5
  br label %bb7

; Loop preheader
; CHECK: vmov.f32
bb7:                                              ; preds = %bb9, %bb.nph
  %s1.02 = phi float [ undef, %bb.nph ], [ %35, %bb9 ] ; <float> [#uses=3]
  %tmp79 = add i32 undef, undef                   ; <i32> [#uses=1]
  %tmp53 = sub i32 undef, undef                   ; <i32> [#uses=1]
  %0 = fadd float 0.000000e+00, 1.000000e+00      ; <float> [#uses=2]
  %1 = fmul float 0.000000e+00, 0.000000e+00      ; <float> [#uses=2]
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7
; CHECK: %bb8
; CHECK-NOT: vmov.f32
; CHECK: blt
  %tmp54 = add i32 0, %tmp53                      ; <i32> [#uses=0]
  %fi.1 = getelementptr float, float* %fz, i32 undef     ; <float*> [#uses=2]
  %tmp80 = add i32 0, %tmp79                      ; <i32> [#uses=1]
  %scevgep81 = getelementptr float, float* %fz, i32 %tmp80 ; <float*> [#uses=1]
  %2 = load float, float* undef, align 4                 ; <float> [#uses=1]
  %3 = fmul float %2, %1                          ; <float> [#uses=1]
  %4 = load float, float* null, align 4                  ; <float> [#uses=2]
  %5 = fmul float %4, %0                          ; <float> [#uses=1]
  %6 = fsub float %3, %5                          ; <float> [#uses=1]
  %7 = fmul float %4, %1                          ; <float> [#uses=1]
  %8 = fadd float undef, %7                       ; <float> [#uses=2]
  %9 = load float, float* %fi.1, align 4                 ; <float> [#uses=2]
  %10 = fsub float %9, %8                         ; <float> [#uses=1]
  %11 = fadd float %9, %8                         ; <float> [#uses=1]
  %12 = fsub float 0.000000e+00, %6               ; <float> [#uses=1]
  %13 = fsub float 0.000000e+00, undef            ; <float> [#uses=2]
  %14 = fmul float undef, %0                      ; <float> [#uses=1]
  %15 = fadd float %14, undef                     ; <float> [#uses=2]
  %16 = load float, float* %scevgep81, align 4           ; <float> [#uses=2]
  %17 = fsub float %16, %15                       ; <float> [#uses=1]
  %18 = fadd float %16, %15                       ; <float> [#uses=2]
  %19 = load float, float* undef, align 4                ; <float> [#uses=2]
  %20 = fsub float %19, %13                       ; <float> [#uses=2]
  %21 = fadd float %19, %13                       ; <float> [#uses=1]
  %22 = fmul float %s1.02, %18                    ; <float> [#uses=1]
  %23 = fmul float 0.000000e+00, %20              ; <float> [#uses=1]
  %24 = fsub float %22, %23                       ; <float> [#uses=1]
  %25 = fmul float 0.000000e+00, %18              ; <float> [#uses=1]
  %26 = fmul float %s1.02, %20                    ; <float> [#uses=1]
  %27 = fadd float %25, %26                       ; <float> [#uses=1]
  %28 = fadd float %11, %27                       ; <float> [#uses=1]
  store float %28, float* %fi.1, align 4
  %29 = fadd float %12, %24                       ; <float> [#uses=1]
  store float %29, float* null, align 4
  %30 = fmul float 0.000000e+00, %21              ; <float> [#uses=1]
  %31 = fmul float %s1.02, %17                    ; <float> [#uses=1]
  %32 = fsub float %30, %31                       ; <float> [#uses=1]
  %33 = fsub float %10, %32                       ; <float> [#uses=1]
  store float %33, float* undef, align 4
  %34 = icmp slt i32 undef, undef                 ; <i1> [#uses=1]
  br i1 %34, label %bb8, label %bb9

bb9:                                              ; preds = %bb8
  %35 = fadd float 0.000000e+00, undef            ; <float> [#uses=1]
  br label %bb7
}
