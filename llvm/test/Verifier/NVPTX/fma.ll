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
