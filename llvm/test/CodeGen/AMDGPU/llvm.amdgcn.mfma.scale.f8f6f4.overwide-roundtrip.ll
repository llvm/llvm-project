; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 < %s | llvm-mc -triple=amdgpu9.50-amd-amdhsa -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 < %s | llvm-mc -triple=amdgpu9.50-amd-amdhsa -filetype=obj -o /dev/null

define <4 x float> @mfma_fp4_overwide(<8 x i32> %a, <8 x i32> %b,
                                      <4 x float> %c) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret <4 x float> %result
}
