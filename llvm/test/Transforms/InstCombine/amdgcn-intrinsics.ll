; RUN: opt -instcombine -S < %s | FileCheck %s

; --------------------------------------------------------------------
; llvm.amdgcn.rcp
; --------------------------------------------------------------------

declare float @llvm.amdgcn.rcp.f32(float) nounwind readnone
declare double @llvm.amdgcn.rcp.f64(double) nounwind readnone


; CHECK-LABEL: @test_constant_fold_rcp_f32_1
; CHECK-NEXT: ret float 1.000000e+00
define float @test_constant_fold_rcp_f32_1() nounwind {
  %val = call float @llvm.amdgcn.rcp.f32(float 1.0) nounwind readnone
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_rcp_f64_1
; CHECK-NEXT:  ret double 1.000000e+00
define double @test_constant_fold_rcp_f64_1() nounwind {
  %val = call double @llvm.amdgcn.rcp.f64(double 1.0) nounwind readnone
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_rcp_f32_half
; CHECK-NEXT: ret float 2.000000e+00
define float @test_constant_fold_rcp_f32_half() nounwind {
  %val = call float @llvm.amdgcn.rcp.f32(float 0.5) nounwind readnone
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_rcp_f64_half
; CHECK-NEXT:  ret double 2.000000e+00
define double @test_constant_fold_rcp_f64_half() nounwind {
  %val = call double @llvm.amdgcn.rcp.f64(double 0.5) nounwind readnone
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_rcp_f32_43
; CHECK-NEXT: call float @llvm.amdgcn.rcp.f32(float 4.300000e+01)
define float @test_constant_fold_rcp_f32_43() nounwind {
 %val = call float @llvm.amdgcn.rcp.f32(float 4.300000e+01) nounwind readnone
 ret float %val
}

; CHECK-LABEL: @test_constant_fold_rcp_f64_43
; CHECK-NEXT: call double @llvm.amdgcn.rcp.f64(double 4.300000e+01)
define double @test_constant_fold_rcp_f64_43() nounwind {
  %val = call double @llvm.amdgcn.rcp.f64(double 4.300000e+01) nounwind readnone
  ret double %val
}


; --------------------------------------------------------------------
; llvm.amdgcn.frexp.mant
; --------------------------------------------------------------------

declare float @llvm.amdgcn.frexp.mant.f32(float) nounwind readnone
declare double @llvm.amdgcn.frexp.mant.f64(double) nounwind readnone


; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_undef(
; CHECK-NEXT: ret float undef
define float @test_constant_fold_frexp_mant_f32_undef() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float undef)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_undef(
; CHECK-NEXT:  ret double undef
define double @test_constant_fold_frexp_mant_f64_undef() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double undef)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_0(
; CHECK-NEXT: ret float 0.000000e+00
define float @test_constant_fold_frexp_mant_f32_0() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float 0.0)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_0(
; CHECK-NEXT:  ret double 0.000000e+00
define double @test_constant_fold_frexp_mant_f64_0() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double 0.0)
  ret double %val
}


; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_n0(
; CHECK-NEXT: ret float -0.000000e+00
define float @test_constant_fold_frexp_mant_f32_n0() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float -0.0)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_n0(
; CHECK-NEXT:  ret double -0.000000e+00
define double @test_constant_fold_frexp_mant_f64_n0() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double -0.0)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_1(
; CHECK-NEXT: ret float 5.000000e-01
define float @test_constant_fold_frexp_mant_f32_1() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float 1.0)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_1(
; CHECK-NEXT:  ret double 5.000000e-01
define double @test_constant_fold_frexp_mant_f64_1() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double 1.0)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_n1(
; CHECK-NEXT: ret float -5.000000e-01
define float @test_constant_fold_frexp_mant_f32_n1() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float -1.0)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_n1(
; CHECK-NEXT:  ret double -5.000000e-01
define double @test_constant_fold_frexp_mant_f64_n1() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double -1.0)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_nan(
; CHECK-NEXT: ret float 0x7FF8000000000000
define float @test_constant_fold_frexp_mant_f32_nan() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float 0x7FF8000000000000)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_nan(
; CHECK-NEXT:  ret double 0x7FF8000000000000
define double @test_constant_fold_frexp_mant_f64_nan() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double 0x7FF8000000000000)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_inf(
; CHECK-NEXT: ret float 0x7FF0000000000000
define float @test_constant_fold_frexp_mant_f32_inf() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float 0x7FF0000000000000)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_inf(
; CHECK-NEXT:  ret double 0x7FF0000000000000
define double @test_constant_fold_frexp_mant_f64_inf() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double 0x7FF0000000000000)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_ninf(
; CHECK-NEXT: ret float 0xFFF0000000000000
define float @test_constant_fold_frexp_mant_f32_ninf() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float 0xFFF0000000000000)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_ninf(
; CHECK-NEXT:  ret double 0xFFF0000000000000
define double @test_constant_fold_frexp_mant_f64_ninf() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double 0xFFF0000000000000)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_max_num(
; CHECK-NEXT: ret float 0x3FEFFFFFE0000000
define float @test_constant_fold_frexp_mant_f32_max_num() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float 0x47EFFFFFE0000000)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_max_num(
; CHECK-NEXT:  ret double 0x3FEFFFFFFFFFFFFF
define double @test_constant_fold_frexp_mant_f64_max_num() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double 0x7FEFFFFFFFFFFFFF)
  ret double %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f32_min_num(
; CHECK-NEXT: ret float 5.000000e-01
define float @test_constant_fold_frexp_mant_f32_min_num() nounwind {
  %val = call float @llvm.amdgcn.frexp.mant.f32(float 0x36A0000000000000)
  ret float %val
}

; CHECK-LABEL: @test_constant_fold_frexp_mant_f64_min_num(
; CHECK-NEXT:  ret double 5.000000e-01
define double @test_constant_fold_frexp_mant_f64_min_num() nounwind {
  %val = call double @llvm.amdgcn.frexp.mant.f64(double 4.940656e-324)
  ret double %val
}


; --------------------------------------------------------------------
; llvm.amdgcn.frexp.exp
; --------------------------------------------------------------------

declare i32 @llvm.amdgcn.frexp.exp.f32(float) nounwind readnone
declare i32 @llvm.amdgcn.frexp.exp.f64(double) nounwind readnone

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_undef(
; CHECK-NEXT: ret i32 undef
define i32 @test_constant_fold_frexp_exp_f32_undef() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float undef)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_undef(
; CHECK-NEXT:  ret i32 undef
define i32 @test_constant_fold_frexp_exp_f64_undef() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double undef)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_0(
; CHECK-NEXT: ret i32 0
define i32 @test_constant_fold_frexp_exp_f32_0() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 0.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_0(
; CHECK-NEXT:  ret i32 0
define i32 @test_constant_fold_frexp_exp_f64_0() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 0.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_n0(
; CHECK-NEXT: ret i32 0
define i32 @test_constant_fold_frexp_exp_f32_n0() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float -0.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_n0(
; CHECK-NEXT:  ret i32 0
define i32 @test_constant_fold_frexp_exp_f64_n0() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double -0.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_1024(
; CHECK-NEXT: ret i32 11
define i32 @test_constant_fold_frexp_exp_f32_1024() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 1024.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_1024(
; CHECK-NEXT:  ret i32 11
define i32 @test_constant_fold_frexp_exp_f64_1024() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 1024.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_n1024(
; CHECK-NEXT: ret i32 11
define i32 @test_constant_fold_frexp_exp_f32_n1024() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float -1024.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_n1024(
; CHECK-NEXT:  ret i32 11
define i32 @test_constant_fold_frexp_exp_f64_n1024() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double -1024.0)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_1_1024(
; CHECK-NEXT: ret i32 -9
define i32 @test_constant_fold_frexp_exp_f32_1_1024() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 0.0009765625)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_1_1024(
; CHECK-NEXT:  ret i32 -9
define i32 @test_constant_fold_frexp_exp_f64_1_1024() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 0.0009765625)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_nan(
; CHECK-NEXT: ret i32 0
define i32 @test_constant_fold_frexp_exp_f32_nan() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 0x7FF8000000000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_nan(
; CHECK-NEXT:  ret i32 0
define i32 @test_constant_fold_frexp_exp_f64_nan() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 0x7FF8000000000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_inf(
; CHECK-NEXT: ret i32 0
define i32 @test_constant_fold_frexp_exp_f32_inf() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 0x7FF0000000000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_inf(
; CHECK-NEXT:  ret i32 0
define i32 @test_constant_fold_frexp_exp_f64_inf() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 0x7FF0000000000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_ninf(
; CHECK-NEXT: ret i32 0
define i32 @test_constant_fold_frexp_exp_f32_ninf() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 0xFFF0000000000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_ninf(
; CHECK-NEXT:  ret i32 0
define i32 @test_constant_fold_frexp_exp_f64_ninf() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 0xFFF0000000000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_max_num(
; CHECK-NEXT: ret i32 128
define i32 @test_constant_fold_frexp_exp_f32_max_num() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 0x47EFFFFFE0000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_max_num(
; CHECK-NEXT:  ret i32 1024
define i32 @test_constant_fold_frexp_exp_f64_max_num() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 0x7FEFFFFFFFFFFFFF)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f32_min_num(
; CHECK-NEXT: ret i32 -148
define i32 @test_constant_fold_frexp_exp_f32_min_num() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f32(float 0x36A0000000000000)
  ret i32 %val
}

; CHECK-LABEL: @test_constant_fold_frexp_exp_f64_min_num(
; CHECK-NEXT:  ret i32 -1073
define i32 @test_constant_fold_frexp_exp_f64_min_num() nounwind {
  %val = call i32 @llvm.amdgcn.frexp.exp.f64(double 4.940656e-324)
  ret i32 %val
}

