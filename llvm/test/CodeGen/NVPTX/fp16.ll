; RUN: llc -mtriple=nvptx64 -verify-machineinstrs < %s | FileCheck %s
; RUN: %if ptxas %{ llc -mtriple=nvptx64 -verify-machineinstrs < %s | %ptxas-verify %}

declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone
declare double @llvm.convert.from.fp16.f64(i16) nounwind readnone
declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone
declare i16 @llvm.convert.to.fp16.f64(double) nounwind readnone

; CHECK-LABEL: @test_convert_fp16_to_fp32
; CHECK: cvt.f32.f16
define void @test_convert_fp16_to_fp32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  %val = load i16, ptr addrspace(1) %in, align 2
  %cvt = call float @llvm.convert.from.fp16.f32(i16 %val) nounwind readnone
  store float %cvt, ptr addrspace(1) %out, align 4
  ret void
}


; CHECK-LABEL: @test_convert_fp16_to_fp64
; CHECK: cvt.f64.f16
define void @test_convert_fp16_to_fp64(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  %val = load i16, ptr addrspace(1) %in, align 2
  %cvt = call double @llvm.convert.from.fp16.f64(i16 %val) nounwind readnone
  store double %cvt, ptr addrspace(1) %out, align 4
  ret void
}


; CHECK-LABEL: @test_convert_fp32_to_fp16
; CHECK: cvt.rn.f16.f32
define void @test_convert_fp32_to_fp16(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  %val = load float, ptr addrspace(1) %in, align 2
  %cvt = call i16 @llvm.convert.to.fp16.f32(float %val) nounwind readnone
  store i16 %cvt, ptr addrspace(1) %out, align 4
  ret void
}


; CHECK-LABEL: @test_convert_fp64_to_fp16
; CHECK: cvt.rn.f16.f64
define void @test_convert_fp64_to_fp16(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  %val = load double, ptr addrspace(1) %in, align 2
  %cvt = call i16 @llvm.convert.to.fp16.f64(double %val) nounwind readnone
  store i16 %cvt, ptr addrspace(1) %out, align 4
  ret void
}
