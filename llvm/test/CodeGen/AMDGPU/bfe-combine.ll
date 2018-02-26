; RUN: llc -march=amdgcn -mcpu=fiji -amdgpu-sdwa-peephole=0 < %s | FileCheck --check-prefix=GCN --check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=fiji < %s | FileCheck --check-prefix=GCN --check-prefix=VI-SDWA %s
; RUN: llc -march=amdgcn -mcpu=bonaire < %s | FileCheck --check-prefix=GCN --check-prefix=CI %s

; GCN-LABEL: {{^}}bfe_combine8:
; VI: v_bfe_u32 v[[BFE:[0-9]+]], v{{[0-9]+}}, 8, 8
; VI: v_lshlrev_b32_e32 v[[ADDRBASE:[0-9]+]], 2, v[[BFE]]
; VI-SDWA: v_mov_b32_e32 v[[SHIFT:[0-9]+]], 2
; VI-SDWA: v_lshlrev_b32_sdwa v[[ADDRBASE:[0-9]+]], v[[SHIFT]], v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
; CI: v_lshrrev_b32_e32 v[[SHR:[0-9]+]], 6, v{{[0-9]+}}
; CI: v_and_b32_e32 v[[ADDRLO:[0-9]+]], 0x3fc, v[[SHR]]
; VI: v_add_u32_e32 v[[ADDRLO:[0-9]+]], vcc, s{{[0-9]+}}, v[[ADDRBASE]]
; VI-SDWA: v_add_u32_e32 v[[ADDRLO:[0-9]+]], vcc, s{{[0-9]+}}, v[[ADDRBASE]]
; GCN: load_dword v{{[0-9]+}}, v{{\[}}[[ADDRLO]]:
define amdgpu_kernel void @bfe_combine8(i32 addrspace(1)* nocapture %arg, i32 %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x() #2
  %idx = add i32 %x, %id
  %srl = lshr i32 %idx, 8
  %and = and i32 %srl, 255
  %ptr = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %and
  %val = load i32, i32 addrspace(1)* %ptr, align 4
  store i32 %val, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}bfe_combine16:
; VI: v_bfe_u32 v[[BFE:[0-9]+]], v{{[0-9]+}}, 16, 16
; VI: v_lshlrev_b32_e32 v[[ADDRBASE:[0-9]+]], {{[^,]+}}, v[[BFE]]
; VI-SDWA: v_mov_b32_e32 v[[SHIFT:[0-9]+]], 15
; VI-SDWA: v_lshlrev_b32_sdwa v[[ADDRBASE1:[0-9]+]], v[[SHIFT]], v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; VI-SDWA: v_lshlrev_b64 v{{\[}}[[ADDRBASE:[0-9]+]]:{{[^\]+}}], 2, v{{\[}}[[ADDRBASE1]]:{{[^\]+}}]
; VI-SDWA: v_add_u32_e32 v[[ADDRLO:[0-9]+]], vcc, s{{[0-9]+}}, v[[ADDRBASE]]
; CI: v_lshrrev_b32_e32 v[[SHR:[0-9]+]], 1, v{{[0-9]+}}
; CI: v_and_b32_e32 v[[AND:[0-9]+]], 0x7fff8000, v[[SHR]]
; CI: v_lshl_b64 v{{\[}}[[ADDRLO:[0-9]+]]:{{[^\]+}}], v{{\[}}[[AND]]:{{[^\]+}}], 2
; VI: v_add_u32_e32 v[[ADDRLO:[0-9]+]], vcc, s{{[0-9]+}}, v[[ADDRBASE]]
; GCN: load_dword v{{[0-9]+}}, v{{\[}}[[ADDRLO]]:
define amdgpu_kernel void @bfe_combine16(i32 addrspace(1)* nocapture %arg, i32 %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x() #2
  %idx = add i32 %x, %id
  %srl = lshr i32 %idx, 1
  %and = and i32 %srl, 2147450880
  %ptr = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %and
  %val = load i32, i32 addrspace(1)* %ptr, align 4
  store i32 %val, i32 addrspace(1)* %arg, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
