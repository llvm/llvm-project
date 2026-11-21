; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=0 < %s | llvm-mc -triple=amdgpu12.50 -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=1 -global-isel-abort=1 < %s | llvm-mc -triple=amdgpu12.50 -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=0 < %s | llvm-mc -triple=amdgpu12.50s -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=1 -global-isel-abort=1 < %s | llvm-mc -triple=amdgpu12.50s -filetype=obj -o /dev/null

define <8 x float> @wmma_fp4_overwide(<16 x i32> %a, <16 x i32> %b,
                                      <8 x float> %c) {
  %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 4, <16 x i32> %a, i32 4, <16 x i32> %b, i16 0, <8 x float> %c)
  ret <8 x float> %result
}
