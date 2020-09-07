; RUN: opt -S -early-cse -earlycse-debug-hash < %s | FileCheck %s

declare float @roundf(float) #0
declare float @llvm.round.f32(float) #0
declare double @round(double) #0
declare double @llvm.round.f64(double) #0

; CHECK-LABEL: @constant_fold_round_f32_01
; CHECK-NEXT: ret float 1.000000e+00
define float @constant_fold_round_f32_01() #0 {
  %x = call float @roundf(float 1.25) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_round_f32_02
; CHECK-NEXT: ret float -1.000000e+00
define float @constant_fold_round_f32_02() #0 {
  %x = call float @llvm.round.f32(float -1.25) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_round_f32_03
; CHECK-NEXT: ret float 2.000000e+00
define float @constant_fold_round_f32_03() #0 {
  %x = call float @roundf(float 1.5) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_round_f32_04
; CHECK-NEXT: ret float -2.000000e+00
define float @constant_fold_round_f32_04() #0 {
  %x = call float @llvm.round.f32(float -1.5) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_round_f32_05
; CHECK-NEXT: ret float 3.000000e+00
define float @constant_fold_round_f32_05() #0 {
  %x = call float @roundf(float 2.75) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_round_f32_06
; CHECK-NEXT: ret float -3.000000e+00
define float @constant_fold_round_f32_06() #0 {
  %x = call float @llvm.round.f32(float -2.75) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_round_f64_01
; CHECK-NEXT: ret double 1.000000e+00
define double @constant_fold_round_f64_01() #0 {
  %x = call double @round(double 1.3) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_round_f64_02
; CHECK-NEXT: ret double -1.000000e+00
define double @constant_fold_round_f64_02() #0 {
  %x = call double @llvm.round.f64(double -1.3) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_round_f64_03
; CHECK-NEXT: ret double 2.000000e+00
define double @constant_fold_round_f64_03() #0 {
  %x = call double @round(double 1.5) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_round_f64_04
; CHECK-NEXT: ret double -2.000000e+00
define double @constant_fold_round_f64_04() #0 {
  %x = call double @llvm.round.f64(double -1.5) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_round_f64_05
; CHECK-NEXT: ret double 3.000000e+00
define double @constant_fold_round_f64_05() #0 {
  %x = call double @round(double 2.7) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_round_f64_06
; CHECK-NEXT: ret double -3.000000e+00
define double @constant_fold_round_f64_06() #0 {
  %x = call double @llvm.round.f64(double -2.7) #0
  ret double %x
}

attributes #0 = { nounwind readnone }
