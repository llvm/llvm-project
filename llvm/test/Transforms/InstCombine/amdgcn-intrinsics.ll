; RUN: opt -instcombine -S < %s | FileCheck %s

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

