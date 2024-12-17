; RUN: llc -march=amdgcn -mcpu=gfx1200 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX12 %s
; RUN: llc -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX12 %s
; RUN: llc -march=amdgcn -mcpu=gfx1251 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1251 %s

; GCN-LABEL: {{^}}mov_dpp64_test:
; GCN: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
; GCN: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
define amdgpu_kernel void @mov_dpp64_test(ptr addrspace(1) %out, i64 %in1) {
  %tmp0 = call i64 @llvm.amdgcn.mov.dpp.i64(i64 %in1, i32 1, i32 1, i32 1, i1 0) #0
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mov_dpp64_row_share_test:
; GFX12-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_share:1 row_mask:0x1 bank_mask:0x1
; GFX1251: v_mov_b64_dpp v[{{[0-9:]+}}], v[{{[0-9:]+}}] row_share:1 row_mask:0x1 bank_mask:0x1
define amdgpu_kernel void @mov_dpp64_row_share_test(ptr addrspace(1) %out, i64 %in1) {
  %tmp0 = call i64 @llvm.amdgcn.mov.dpp.i64(i64 %in1, i32 337, i32 1, i32 1, i1 0) #0
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

declare i64 @llvm.amdgcn.mov.dpp.i64(i64, i32, i32, i32, i1) #0

attributes #0 = { nounwind readnone convergent }
