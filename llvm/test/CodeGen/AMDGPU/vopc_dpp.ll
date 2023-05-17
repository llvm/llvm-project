; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX11 %s

define amdgpu_cs void @_amdgpu_cs_main(i32 %0) {
; GFX11-LABEL: _amdgpu_cs_main:
; GFX11:    v_cmp_eq_u32_e64_dpp s1, v1, v0 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1
.entry:
  %1 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 0, i32 0, i32 15, i32 15, i1 false)
  %2 = icmp ne i32 %1, %0
  %spec.select.3 = select i1 %2, i32 0, i32 1
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 %spec.select.3, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  ret void
}

declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)
declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32 immarg)
