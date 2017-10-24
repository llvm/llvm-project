; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=fiji -stop-after=irtranslator -global-isel %s -o - | FileCheck %s


; CHECK-LABEL: name: test_f32_inreg
; CHECK: [[S0:%[0-9]+]]:_(s32) = COPY %sgpr0
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), %{{[0-9]+}}(s32), %{{[0-9]+}}(s32), [[S0]]
define amdgpu_vs void @test_f32_inreg(float inreg %arg0) {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg0, float undef, float undef, float undef, i1 false, i1 false) #0
  ret void
}

; CHECK-LABEL: name: test_f32
; CHECK: [[V0:%[0-9]+]]:_(s32) = COPY %vgpr0
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), %{{[0-9]+}}(s32), %{{[0-9]+}}(s32), [[V0]]
define amdgpu_vs void @test_f32(float %arg0) {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg0, float undef, float undef, float undef, i1 false, i1 false) #0
  ret void
}

; CHECK-LABEL: name: test_ptr2_byval
; CHECK: [[S01:%[0-9]+]]:_(p2) = COPY %sgpr0_sgpr1
; CHECK: G_LOAD [[S01]]
define amdgpu_vs void @test_ptr2_byval(i32 addrspace(2)* byval %arg0) {
   %tmp0 = load volatile i32, i32 addrspace(2)* %arg0
  ret void
}

; CHECK-LABEL: name: test_ptr2_inreg
; CHECK: [[S01:%[0-9]+]]:_(p2) = COPY %sgpr0_sgpr1
; CHECK: G_LOAD [[S01]]
define amdgpu_vs void @test_ptr2_inreg(i32 addrspace(2)* inreg %arg0) {
  %tmp0 = load volatile i32, i32 addrspace(2)* %arg0
  ret void
}

; CHECK-LABEL: name: test_sgpr_alignment0
; CHECK: [[S0:%[0-9]+]]:_(s32) = COPY %sgpr0
; CHECK: [[S23:%[0-9]+]]:_(p2) = COPY %sgpr2_sgpr3
; CHECK: G_LOAD [[S23]]
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), %{{[0-9]+}}(s32), %{{[0-9]+}}(s32), [[S0]]
define amdgpu_vs void @test_sgpr_alignment0(float inreg %arg0, i32 addrspace(2)* inreg %arg1) {
  %tmp0 = load volatile i32, i32 addrspace(2)* %arg1
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg0, float undef, float undef, float undef, i1 false, i1 false) #0
  ret void
}

; CHECK-LABEL: name: test_order
; CHECK: [[S0:%[0-9]+]]:_(s32) = COPY %sgpr0
; CHECK: [[S1:%[0-9]+]]:_(s32) = COPY %sgpr1
; CHECK: [[V0:%[0-9]+]]:_(s32) = COPY %vgpr0
; CHECK: [[V1:%[0-9]+]]:_(s32) = COPY %vgpr1
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), %{{[0-9]+}}(s32), %{{[0-9]+}}(s32), [[V0]](s32), [[S0]](s32), [[V1]](s32), [[S1]](s32)
define amdgpu_vs void @test_order(float inreg %arg0, float inreg %arg1, float %arg2, float %arg3) {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg2, float %arg0, float %arg3, float %arg1, i1 false, i1 false) #0
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)  #0

attributes #0 = { nounwind }
