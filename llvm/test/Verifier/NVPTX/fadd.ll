; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare float @llvm.nvvm.fadd.f32(float, float, i32 immarg)

define void @test_fadd_rounding_mode(float %a) {
  ; CHECK: immarg value 4 for arg 2 out of range [0,4)
  call float @llvm.nvvm.fadd.f32(float %a, float %a, i32 4)

  ; CHECK: immarg value 7 for arg 2 out of range [0,4)
  call float @llvm.nvvm.fadd.f32(float %a, float %a, i32 7)

  ; CHECK: immarg value -1 for arg 2 out of range [0,4)
  call float @llvm.nvvm.fadd.f32(float %a, float %a, i32 -1)

  ret void
}
