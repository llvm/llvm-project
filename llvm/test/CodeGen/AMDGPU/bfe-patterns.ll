; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}v_ubfe_sub_i32:
; GCN: {{buffer|flat}}_load_dword [[SRC:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[WIDTH:v[0-9]+]]
; GCN: v_bfe_u32 v{{[0-9]+}}, [[SRC]], 0, [[WIDTH]]
define amdgpu_kernel void @v_ubfe_sub_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in0.gep = getelementptr i32, i32 addrspace(1)* %in0, i32 %id.x
  %in1.gep = getelementptr i32, i32 addrspace(1)* %in1, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %src = load volatile i32, i32 addrspace(1)* %in0.gep
  %width = load volatile i32, i32 addrspace(1)* %in0.gep
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = lshr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_ubfe_sub_multi_use_shl_i32:
; GCN: {{buffer|flat}}_load_dword [[SRC:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[WIDTH:v[0-9]+]]
; GCN: v_sub_{{[iu]}}32_e32 [[SUB:v[0-9]+]], vcc, 32, [[WIDTH]]

; SI-NEXT: v_lshl_b32_e32 [[SHL:v[0-9]+]], [[SRC]], [[SUB]]
; SI-NEXT: v_lshr_b32_e32 [[BFE:v[0-9]+]], [[SHL]], [[SUB]]

; VI-NEXT: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], [[SUB]], [[SRC]]
; VI-NEXT: v_lshrrev_b32_e32 [[BFE:v[0-9]+]], [[SUB]], [[SHL]]

; GCN: [[BFE]]
; GCN: [[SHL]]
define amdgpu_kernel void @v_ubfe_sub_multi_use_shl_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in0.gep = getelementptr i32, i32 addrspace(1)* %in0, i32 %id.x
  %in1.gep = getelementptr i32, i32 addrspace(1)* %in1, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %src = load volatile i32, i32 addrspace(1)* %in0.gep
  %width = load volatile i32, i32 addrspace(1)* %in0.gep
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = lshr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  store volatile i32 %shl, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}s_ubfe_sub_i32:
; GCN: s_load_dwordx2 s{{\[}}[[SRC:[0-9]+]]:[[WIDTH:[0-9]+]]{{\]}}, s[0:1], {{0xb|0x2c}}
; GCN: v_mov_b32_e32 [[VWIDTH:v[0-9]+]], s[[WIDTH]]
; GCN: v_bfe_u32 v{{[0-9]+}}, s[[SRC]], 0, [[VWIDTH]]
define amdgpu_kernel void @s_ubfe_sub_i32(i32 addrspace(1)* %out, i32 %src, i32 %width) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = lshr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}s_ubfe_sub_multi_use_shl_i32:
; GCN: s_load_dwordx2 s{{\[}}[[SRC:[0-9]+]]:[[WIDTH:[0-9]+]]{{\]}}, s[0:1], {{0xb|0x2c}}
; GCN: s_sub_i32 [[SUB:s[0-9]+]], 32, s[[WIDTH]]
; GCN: s_lshl_b32 [[SHL:s[0-9]+]], s[[SRC]], [[SUB]]
; GCN: s_lshr_b32 s{{[0-9]+}}, [[SHL]], [[SUB]]
define amdgpu_kernel void @s_ubfe_sub_multi_use_shl_i32(i32 addrspace(1)* %out, i32 %src, i32 %width) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = lshr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  store volatile i32 %shl, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_sbfe_sub_i32:
; GCN: {{buffer|flat}}_load_dword [[SRC:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[WIDTH:v[0-9]+]]
; GCN: v_bfe_i32 v{{[0-9]+}}, [[SRC]], 0, [[WIDTH]]
define amdgpu_kernel void @v_sbfe_sub_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in0.gep = getelementptr i32, i32 addrspace(1)* %in0, i32 %id.x
  %in1.gep = getelementptr i32, i32 addrspace(1)* %in1, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %src = load volatile i32, i32 addrspace(1)* %in0.gep
  %width = load volatile i32, i32 addrspace(1)* %in0.gep
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = ashr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_sbfe_sub_multi_use_shl_i32:
; GCN: {{buffer|flat}}_load_dword [[SRC:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[WIDTH:v[0-9]+]]
; GCN: v_sub_{{[iu]}}32_e32 [[SUB:v[0-9]+]], vcc, 32, [[WIDTH]]

; SI-NEXT: v_lshl_b32_e32 [[SHL:v[0-9]+]], [[SRC]], [[SUB]]
; SI-NEXT: v_ashr_i32_e32 [[BFE:v[0-9]+]], [[SHL]], [[SUB]]

; VI-NEXT: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], [[SUB]], [[SRC]]
; VI-NEXT: v_ashrrev_i32_e32 [[BFE:v[0-9]+]], [[SUB]], [[SHL]]

; GCN: [[BFE]]
; GCN: [[SHL]]
define amdgpu_kernel void @v_sbfe_sub_multi_use_shl_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in0.gep = getelementptr i32, i32 addrspace(1)* %in0, i32 %id.x
  %in1.gep = getelementptr i32, i32 addrspace(1)* %in1, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %src = load volatile i32, i32 addrspace(1)* %in0.gep
  %width = load volatile i32, i32 addrspace(1)* %in0.gep
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = ashr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  store volatile i32 %shl, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}s_sbfe_sub_i32:
; GCN: s_load_dwordx2 s{{\[}}[[SRC:[0-9]+]]:[[WIDTH:[0-9]+]]{{\]}}, s[0:1], {{0xb|0x2c}}
; GCN: v_mov_b32_e32 [[VWIDTH:v[0-9]+]], s[[WIDTH]]
; GCN: v_bfe_i32 v{{[0-9]+}}, s[[SRC]], 0, [[VWIDTH]]
define amdgpu_kernel void @s_sbfe_sub_i32(i32 addrspace(1)* %out, i32 %src, i32 %width) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = ashr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}s_sbfe_sub_multi_use_shl_i32:
; GCN: s_load_dwordx2 s{{\[}}[[SRC:[0-9]+]]:[[WIDTH:[0-9]+]]{{\]}}, s[0:1], {{0xb|0x2c}}
; GCN: s_sub_i32 [[SUB:s[0-9]+]], 32, s[[WIDTH]]
; GCN: s_lshl_b32 [[SHL:s[0-9]+]], s[[SRC]], [[SUB]]
; GCN: s_ashr_i32 s{{[0-9]+}}, [[SHL]], [[SUB]]
define amdgpu_kernel void @s_sbfe_sub_multi_use_shl_i32(i32 addrspace(1)* %out, i32 %src, i32 %width) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %sub = sub i32 32, %width
  %shl = shl i32 %src, %sub
  %bfe = ashr i32 %shl, %sub
  store i32 %bfe, i32 addrspace(1)* %out.gep
  store volatile i32 %shl, i32 addrspace(1)* undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
