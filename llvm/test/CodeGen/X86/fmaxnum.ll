; RUN: llc  -march=x86 -mtriple=i386-linux-gnu  < %s | FileCheck %s

declare float @fmaxf(float, float)
declare double @fmax(double, double)
declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80)

; CHECK-LABEL: @test_fmaxf
; CHECK: calll fmaxf
define float @test_fmaxf(float %x, float %y) {
  %z = call float @fmaxf(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_fmax
; CHECK: calll fmax
define double @test_fmax(double %x, double %y) {
  %z = call double @fmax(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_fmaxl
; CHECK: calll fmaxl
define x86_fp80 @test_fmaxl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @fmaxl(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fmaxf
; CHECK: calll fmaxf
define float @test_intrinsic_fmaxf(float %x, float %y) {
  %z = call float @llvm.maxnum.f32(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_intrinsic_fmax
; CHECK: calll fmax
define double @test_intrinsic_fmax(double %x, double %y) {
  %z = call double @llvm.maxnum.f64(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_intrinsic_fmaxl
; CHECK: calll fmaxl
define x86_fp80 @test_intrinsic_fmaxl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @llvm.maxnum.f80(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}
