; RUN: not --crash llc -global-isel=0 -march=amdgcn -mcpu=fiji -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck -check-prefix=SDAG-ERR %s
; RUN: not --crash llc -global-isel=0 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck -check-prefix=SDAG-ERR %s

; RUN: not --crash llc -global-isel=1 -march=amdgcn -mcpu=fiji -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s
; RUN: not --crash llc -global-isel=1 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s

; Make sure this doesn't assert on targets without the g16 feature, and instead
; generates a selection error.

; SDAG-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.image.sample.d.1d
; GISEL-ERR: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(<4 x s32>) = G_AMDGPU_INTRIN_IMAGE_LOAD intrinsic(@llvm.amdgcn.image.sample.d.1d), 15, %{{[0-9]+}}:_(s16), %{{[0-9]+}}:_(s16), %{{[0-9]+}}:_(s32), %{{[0-9]+}}:_(<8 x s32>), %{{[0-9]+}}:_(<4 x s32>), 0, 0, 0 :: (dereferenceable load 16 from custom "TargetCustom8") (in function: sample_d_1d)

define amdgpu_ps <4 x float> @sample_d_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f16.f32(i32 15, half %dsdh, half %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f16.f32(i32, half, half, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

attributes #0 = { nounwind readonly }
