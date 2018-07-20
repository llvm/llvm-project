; RUN: llc -mtriple=amdgcn-mesa-mesa3d -stop-after=irtranslator -global-isel %s -o - | FileCheck %s

; Check that we correctly skip over disabled inputs
; CHECK: [[S0:%[0-9]+]]:_(s32) = COPY $sgpr0
; CHECK: [[V0:%[0-9]+]]:_(s32) = COPY $vgpr0
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), %{{[0-9]+}}(s32), %{{[0-9]+}}(s32), [[S0]](s32), [[S0]](s32), [[S0]](s32), [[V0]](s32)
define amdgpu_ps void @ps0(float inreg %arg0, float %psinput0, float %psinput1) #1 {
main_body:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %arg0, float %arg0, float %arg0, float %psinput1, i1 false, i1 false) #0
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)  #0

attributes #0 = { nounwind }
attributes #1 = { "InitialPSInputAddr"="0x00002" }
