; RUN: llc -mcpu=cortex-a8 -mattr=-neonfp -arm-atomic-cfg-tidy=0 < %s | FileCheck %s
; PR5423

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

define arm_aapcs_vfpcc void @foo() {
entry:
  %0 = load float, float* null, align 4                  ; <float> [#uses=2]
  %1 = fmul float %0, undef                       ; <float> [#uses=2]
  %2 = fmul float 0.000000e+00, %1                ; <float> [#uses=2]
  %3 = fmul float %0, %1                          ; <float> [#uses=1]
  %4 = fadd float 0.000000e+00, %3                ; <float> [#uses=1]
  %5 = fsub float 1.000000e+00, %4                ; <float> [#uses=1]
; CHECK-LABEL: foo:
; CHECK: vmov.f32 s{{[0-9]+}}, #1.000000e+00
  %6 = fsub float 1.000000e+00, undef             ; <float> [#uses=2]
  %7 = fsub float %2, undef                       ; <float> [#uses=1]
  %8 = fsub float 0.000000e+00, undef             ; <float> [#uses=3]
  %9 = fadd float %2, undef                       ; <float> [#uses=3]
  %10 = load float, float* undef, align 8                ; <float> [#uses=3]
  %11 = fmul float %8, %10                        ; <float> [#uses=1]
  %12 = fadd float undef, %11                     ; <float> [#uses=2]
  %13 = fmul float undef, undef                   ; <float> [#uses=1]
  %14 = fmul float %6, 0.000000e+00               ; <float> [#uses=1]
  %15 = fadd float %13, %14                       ; <float> [#uses=1]
  %16 = fmul float %9, %10                        ; <float> [#uses=1]
  %17 = fadd float %15, %16                       ; <float> [#uses=2]
  %18 = fmul float 0.000000e+00, undef            ; <float> [#uses=1]
  %19 = fadd float %18, 0.000000e+00              ; <float> [#uses=1]
  %20 = fmul float undef, %10                     ; <float> [#uses=1]
  %21 = fadd float %19, %20                       ; <float> [#uses=1]
  %22 = load float, float* undef, align 8                ; <float> [#uses=1]
  %23 = fmul float %5, %22                        ; <float> [#uses=1]
  %24 = fadd float %23, undef                     ; <float> [#uses=1]
  %25 = load float, float* undef, align 8                ; <float> [#uses=2]
  %26 = fmul float %8, %25                        ; <float> [#uses=1]
  %27 = fadd float %24, %26                       ; <float> [#uses=1]
  %28 = fmul float %9, %25                        ; <float> [#uses=1]
  %29 = fadd float undef, %28                     ; <float> [#uses=1]
  %30 = fmul float %8, undef                      ; <float> [#uses=1]
  %31 = fadd float undef, %30                     ; <float> [#uses=1]
  %32 = fmul float %6, undef                      ; <float> [#uses=1]
  %33 = fadd float undef, %32                     ; <float> [#uses=1]
  %34 = fmul float %9, undef                      ; <float> [#uses=1]
  %35 = fadd float %33, %34                       ; <float> [#uses=1]
  %36 = fmul float 0.000000e+00, undef            ; <float> [#uses=1]
  %37 = fmul float %7, undef                      ; <float> [#uses=1]
  %38 = fadd float %36, %37                       ; <float> [#uses=1]
  %39 = fmul float undef, undef                   ; <float> [#uses=1]
  %40 = fadd float %38, %39                       ; <float> [#uses=1]
  store float %12, float* undef, align 8
  store float %17, float* undef, align 4
  store float %21, float* undef, align 8
  store float %27, float* undef, align 8
  store float %29, float* undef, align 4
  store float %31, float* undef, align 8
  store float %40, float* undef, align 8
  store float %12, float* null, align 8
  %41 = fmul float %17, undef                     ; <float> [#uses=1]
  %42 = fadd float %41, undef                     ; <float> [#uses=1]
  %43 = fmul float %35, undef                     ; <float> [#uses=1]
  %44 = fadd float %42, %43                       ; <float> [#uses=1]
  store float %44, float* null, align 4
  unreachable
}
