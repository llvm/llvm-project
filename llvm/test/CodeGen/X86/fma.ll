; RUN: llc < %s -mtriple=i386-apple-darwin10  -mattr=+fma,-fma4  | FileCheck %s --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -mtriple=i386-apple-darwin10  -mattr=-fma,-fma4  | FileCheck %s --check-prefix=CHECK-FMA-CALL
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -mattr=+fma,-fma4 | FileCheck %s --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -mtriple=x86_64-apple-darwin10  -mattr=-fma,-fma4 | FileCheck %s --check-prefix=CHECK-FMA-CALL
; RUN: llc < %s -march=x86 -mcpu=bdver2 -mattr=-fma4  | FileCheck %s --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -march=x86 -mcpu=bdver2 -mattr=-fma,-fma4 | FileCheck %s --check-prefix=CHECK-FMA-CALL

; CHECK: test_f32
; CHECK-FMA-INST: vfmadd213ss
; CHECK-FMA-CALL: fmaf

define float @test_f32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
  %call = tail call float @llvm.fma.f32(float %a, float %b, float %c) nounwind readnone
  ret float %call
}

; CHECK: test_f64
; CHECK-FMA-INST: vfmadd213sd
; CHECK-FMA-CALL: fma

define double @test_f64(double %a, double %b, double %c) nounwind readnone ssp {
entry:
  %call = tail call double @llvm.fma.f64(double %a, double %b, double %c) nounwind readnone
  ret double %call
}

; CHECK: test_f80
; CHECK: fmal

define x86_fp80 @test_f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c) nounwind readnone ssp {
entry:
  %call = tail call x86_fp80 @llvm.fma.f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c) nounwind readnone
  ret x86_fp80 %call
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare double @llvm.fma.f64(double, double, double) nounwind readnone
declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) nounwind readnone
