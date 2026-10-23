; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare float @llvm.nvvm.fmul.f32(float, float, i32 immarg)

define void @test_fmul_rounding_mode(float %a) {
  ; CHECK: immarg value 4 for arg 2 out of range [0,4)
  call float @llvm.nvvm.fmul.f32(float %a, float %a, i32 4)

  ; CHECK: immarg value 7 for arg 2 out of range [0,4)
  call float @llvm.nvvm.fmul.f32(float %a, float %a, i32 7)

  ; CHECK: immarg value -1 for arg 2 out of range [0,4)
  call float @llvm.nvvm.fmul.f32(float %a, float %a, i32 -1)

  ret void
}

define void @test_fmul_unsupported(half %h, bfloat %bf, <2 x float> %v2f32, <4 x float> %v4f32, double %d, <2 x double> %v2f64, fp128 %q) {
  ; CHECK: llvm.nvvm.fmul.sat with rounding mode rz and operand type half is not supported
  call half @llvm.nvvm.fmul.sat.f16(half %h, half %h, i32 0)

  ; CHECK: llvm.nvvm.fmul.ftz with rounding mode rn and operand type bfloat is not supported
  call bfloat @llvm.nvvm.fmul.ftz.bf16(bfloat %bf, bfloat %bf, i32 1)

  ; CHECK: llvm.nvvm.fmul with rounding mode rp and operand type bfloat is not supported
  call bfloat @llvm.nvvm.fmul.bf16(bfloat %bf, bfloat %bf, i32 2)

  ; CHECK: llvm.nvvm.fmul.sat with rounding mode rn and operand type <2 x float> is not supported
  call <2 x float> @llvm.nvvm.fmul.sat.v2f32(<2 x float> %v2f32, <2 x float> %v2f32, i32 1)

  ; CHECK: llvm.nvvm.fmul with rounding mode rn and operand type <4 x float> is not supported
  call <4 x float> @llvm.nvvm.fmul.v4f32(<4 x float> %v4f32, <4 x float> %v4f32, i32 1)

  ; CHECK: llvm.nvvm.fmul.ftz with rounding mode rn and operand type double is not supported
  call double @llvm.nvvm.fmul.ftz.f64(double %d, double %d, i32 1)

  ; CHECK: llvm.nvvm.fmul with rounding mode rn and operand type <2 x double> is not supported
  call <2 x double> @llvm.nvvm.fmul.v2f64(<2 x double> %v2f64, <2 x double> %v2f64, i32 1)

  ; CHECK: llvm.nvvm.fmul with rounding mode rn and operand type fp128 is not supported
  call fp128 @llvm.nvvm.fmul.f128(fp128 %q, fp128 %q, i32 1)

  ret void
}
