; RUN: not --crash llc -march=amdgcn -mcpu=fiji -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck -check-prefix=ERR %s

; Make sure this doesn't assert on targets without the r128-16
; feature, and instead generates a slection error.

; ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.image.load.1d

define amdgpu_ps <4 x float> @load_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #0

attributes #0 = { nounwind readonly }
