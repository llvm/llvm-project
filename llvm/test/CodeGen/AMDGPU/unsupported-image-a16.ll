; RUN: not --crash llc -global-isel=0 -march=amdgcn -mcpu=fiji -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck -check-prefix=SDAG-ERR %s
; RUN: not --crash llc -global-isel=1 -march=amdgcn -mcpu=fiji -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s

; Make sure this doesn't assert on targets without the r128-16
; feature, and instead generates a slection error.

; SDAG-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.image.load.1d
; GISEL-ERR: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(<4 x s32>) = G_AMDGPU_INTRIN_IMAGE_LOAD intrinsic(@llvm.amdgcn.image.load.1d), 15, %{{[0-9]+}}:_(s16), %{{[0-9]+}}:_(<8 x s32>), 0, 0 :: (dereferenceable load 16 from custom "TargetCustom8") (in function: load_1d)

define amdgpu_ps <4 x float> @load_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #0

attributes #0 = { nounwind readonly }
