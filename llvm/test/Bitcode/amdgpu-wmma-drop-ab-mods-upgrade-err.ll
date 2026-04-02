; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: wmma.f32.16x16x32.bf16.v8f32.v16bf16 matrix A and B modifiers shall be zero
define amdgpu_ps void @test_wmma_f32_16x16x32_bf16_negA(<16 x bfloat> %A, <16 x bfloat> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(i1 true, <16 x bfloat> %A, i1 0, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 true)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}
