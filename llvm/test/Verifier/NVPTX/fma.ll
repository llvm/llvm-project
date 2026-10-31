; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare float @llvm.nvvm.fma.f32(float, float, float, i32 immarg)

define void @test_fma_rounding_mode(float %a) {
  ; CHECK: immarg value 4 for arg 3 out of range [0,4)
  call float @llvm.nvvm.fma.f32(float %a, float %a, float %a, i32 4)

  ; CHECK: immarg value 7 for arg 3 out of range [0,4)
  call float @llvm.nvvm.fma.f32(float %a, float %a, float %a, i32 7)

  ; CHECK: immarg value -1 for arg 3 out of range [0,4)
  call float @llvm.nvvm.fma.f32(float %a, float %a, float %a, i32 -1)

  ret void
}

define void @test_fma_unsupported(half %h, bfloat %bf, float %f, <2 x float> %v2f32, <4 x float> %v4f32, double %d, <2 x double> %v2f64, fp128 %q) {
  ; CHECK: llvm.nvvm.fma.sat with rounding mode rz and operand type half is not supported
  call half @llvm.nvvm.fma.sat.f16(half %h, half %h, half %h, i32 0)

  ; CHECK: llvm.nvvm.fma.relu with rounding mode rz and operand type half is not supported
  call half @llvm.nvvm.fma.relu.f16(half %h, half %h, half %h, i32 0)

  ; CHECK: llvm.nvvm.fma.ftz with rounding mode rn and operand type bfloat is not supported
  call bfloat @llvm.nvvm.fma.ftz.bf16(bfloat %bf, bfloat %bf, bfloat %bf, i32 1)

  ; CHECK: llvm.nvvm.fma.sat with rounding mode rn and operand type bfloat is not supported
  call bfloat @llvm.nvvm.fma.sat.bf16(bfloat %bf, bfloat %bf, bfloat %bf, i32 1)

  ; CHECK: llvm.nvvm.fma.relu with rounding mode rn and operand type float is not supported
  call float @llvm.nvvm.fma.relu.f32(float %f, float %f, float %f, i32 1)

  ; CHECK: llvm.nvvm.fma.sat with rounding mode rn and operand type <2 x float> is not supported
  call <2 x float> @llvm.nvvm.fma.sat.v2f32(<2 x float> %v2f32, <2 x float> %v2f32, <2 x float> %v2f32, i32 1)

  ; CHECK: llvm.nvvm.fma with rounding mode rn and operand type <4 x float> is not supported
  call <4 x float> @llvm.nvvm.fma.v4f32(<4 x float> %v4f32, <4 x float> %v4f32, <4 x float> %v4f32, i32 1)

  ; CHECK: llvm.nvvm.fma.ftz with rounding mode rn and operand type double is not supported
  call double @llvm.nvvm.fma.ftz.f64(double %d, double %d, double %d, i32 1)

  ; CHECK: llvm.nvvm.fma.relu with rounding mode rn and operand type double is not supported
  call double @llvm.nvvm.fma.relu.f64(double %d, double %d, double %d, i32 1)

  ; CHECK: llvm.nvvm.fma with rounding mode rn and operand type <2 x double> is not supported
  call <2 x double> @llvm.nvvm.fma.v2f64(<2 x double> %v2f64, <2 x double> %v2f64, <2 x double> %v2f64, i32 1)

  ; CHECK: llvm.nvvm.fma with rounding mode rn and operand type fp128 is not supported
  call fp128 @llvm.nvvm.fma.f128(fp128 %q, fp128 %q, fp128 %q, i32 1)

  ret void
}
