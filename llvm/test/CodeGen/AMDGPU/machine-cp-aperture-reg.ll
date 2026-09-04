; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -global-isel < %s | FileCheck %s

; src_shared_base has no addressable high half. Must not forward into VGPRs.

@lds = internal addrspace(3) global [32 x float] poison, align 4

; CHECK-LABEL: aperture_copy_to_vgpr_pair:
; CHECK: s_mov_b64 s[{{[0-9]+}}:{{[0-9]+}}], src_shared_base
define float @aperture_copy_to_vgpr_pair(i32 %i) {
  %idx = zext i32 %i to i64
  %gep = getelementptr float, ptr addrspacecast (ptr addrspace(3) @lds to ptr), i64 %idx
  %cast = addrspacecast ptr %gep to ptr addrspace(3)
  %v = load float, ptr addrspace(3) %cast, align 4
  ret float %v
}
