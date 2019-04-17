; RUN: opt -S -instcombine < %s | FileCheck %s

declare float @llvm.copysign.f32(float, float) #0
declare double @llvm.copysign.f64(double, double) #0

; CHECK-LABEL: @constant_fold_copysign_f32_01
; CHECK-NEXT: ret float -1.000000e+00
define float @constant_fold_copysign_f32_01() #0 {
  %x = call float @llvm.copysign.f32(float 1.0, float -2.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_copysign_f32_02
; CHECK-NEXT: ret float 2.000000e+00
define float @constant_fold_copysign_f32_02() #0 {
  %x = call float @llvm.copysign.f32(float -2.0, float 1.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_copysign_f32_03
; CHECK-NEXT: ret float -2.000000e+00
define float @constant_fold_copysign_f32_03() #0 {
  %x = call float @llvm.copysign.f32(float -2.0, float -1.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_copysign_f64_01
; CHECK-NEXT: ret double -1.000000e+00
define double @constant_fold_copysign_f64_01() #0 {
  %x = call double @llvm.copysign.f64(double 1.0, double -2.0) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_copysign_f64_02
; CHECK-NEXT: ret double 1.000000e+00
define double @constant_fold_copysign_f64_02() #0 {
  %x = call double @llvm.copysign.f64(double -1.0, double 2.0) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_copysign_f64_03
; CHECK-NEXT: ret double -1.000000e+00
define double @constant_fold_copysign_f64_03() #0 {
  %x = call double @llvm.copysign.f64(double -1.0, double -2.0) #0
  ret double %x
}


attributes #0 = { nounwind readnone }
