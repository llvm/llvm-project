; RUN: not llc -mtriple=amdgcn -mcpu=gfx1250 < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: error: invalid matrix and scale format combination in wmma call
; ERR: llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
define amdgpu_ps void @test_wmma_scale_f32_16x16x128_f8f6f4(<16 x i32> %A, <16 x i32> %B, <8 x float> %C, i32 inreg %scale_src0, i32 inreg %scale_src1, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 1, <16 x i32> %A, i32 2, <16 x i32> %B, i16 0, <8 x float> %C, i32 2, i32 1, i32 %scale_src0, i32 1, i32 2, i32 %scale_src1, i1 true, i1 false)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; ERR: error: invalid matrix and scale format combination in wmma call
; ERR: llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4
define amdgpu_ps void @test_wmma_scale16_f32_16x16x128_f8f6f4(<16 x i32> %A, <16 x i32> %B, <8 x float> %C, i64 inreg %scale_src0, ptr addrspace(1) %out) {
bb:
 %res = call <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32i.v16i32(i32 1, <16 x i32> %A, i32 2, <16 x i32> %B, i16 0, <8 x float> %C, i32 3, i32 2, i64 %scale_src0, i32 0, i32 1, i64 100, i1 false, i1 true)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

