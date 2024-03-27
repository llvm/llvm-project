; XFAIL: *
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1010 -verify-machineinstrs -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: _amdgpu_hs_main:

define amdgpu_hs void @_amdgpu_hs_main() #0 {
.entry:
  ret void
}

; GCN-LABEL: _amdgpu_ps_main:
; GCN: s_and_saveexec_b64

define amdgpu_ps void @_amdgpu_ps_main(i32 %arg) local_unnamed_addr #1 {
.entry:
  %tmp = tail call float @llvm.amdgcn.interp.p2(float undef, float undef, i32 1, i32 0, i32 %arg) #2
  %tmp1 = tail call float @llvm.amdgcn.image.sample.2d.f32.f32(i32 1, float undef, float %tmp, <8 x i32> undef, <4 x i32> undef, i1 false, i32 0, i32 0)
  %tmp2 = fcmp olt float %tmp1, 5.000000e-01
  br i1 %tmp2, label %bb, label %l

bb:                                               ; preds = %.entry
  unreachable

l: ; preds = %.entry
  ret void
}

; GCN-LABEL: _amdgpu_gs_main:

define amdgpu_gs void @_amdgpu_gs_main() #4 {
.entry:
  ret void
}

declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #2
declare float @llvm.amdgcn.image.sample.2d.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3

attributes #0 = { "amdgpu-max-work-group-size"="128" "target-features"=",+wavefrontsize32" }
attributes #1 = { "target-features"=",+wavefrontsize64" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind readonly }
attributes #4 = { "target-features"=",+wavefrontsize32" }
