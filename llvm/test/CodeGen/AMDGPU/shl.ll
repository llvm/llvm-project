; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s
; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=SI %s
; XUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare i32 @llvm.r600.read.tidig.x() #0


;EG: {{^}}shl_v2i32:
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: {{^}}shl_v2i32:
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

;VI: {{^}}shl_v2i32:
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @shl_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1) * %in
  %b = load <2 x i32>, <2 x i32> addrspace(1) * %b_ptr
  %result = shl <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG: {{^}}shl_v4i32:
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: {{^}}shl_v4i32:
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

;VI: {{^}}shl_v4i32:
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @shl_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1) * %in
  %b = load <4 x i32>, <4 x i32> addrspace(1) * %b_ptr
  %result = shl <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

;EG: {{^}}shl_i64:
;EG: SUB_INT {{\*? *}}[[COMPSH:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHIFT:T[0-9]+\.[XYZW]]]
;EG: LSHR {{\* *}}[[TEMP:T[0-9]+\.[XYZW]]], [[OPLO:T[0-9]+\.[XYZW]]], {{[[COMPSH]]|PV.[XYZW]}}
;EG: LSHR {{\*? *}}[[OVERF:T[0-9]+\.[XYZW]]], {{[[TEMP]]|PV.[XYZW]}}, 1
;EG_CHECK-DAG: ADD_INT {{\*? *}}[[BIGSH:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-DAG: LSHL {{\*? *}}[[HISMTMP:T[0-9]+\.[XYZW]]], [[OPHI:T[0-9]+\.[XYZW]]], [[SHIFT]]
;EG-DAG: OR_INT {{\*? *}}[[HISM:T[0-9]+\.[XYZW]]], {{[[HISMTMP]]|PV.[XYZW]}}, {{[[OVERF]]|PV.[XYZW]}}
;EG-DAG: LSHL {{\*? *}}[[LOSM:T[0-9]+\.[XYZW]]], [[OPLO]], {{PS|[[SHIFT]]}}
;EG-DAG: SETGT_UINT {{\*? *}}[[RESC:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-DAG: CNDE_INT {{\*? *}}[[RESLO:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}
;EG-DAG: CNDE_INT {{\*? *}}[[RESHI:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW], .*}}, 0.0

;SI: {{^}}shl_i64:
;SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI: {{^}}shl_i64:
;VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @shl_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 1
  %a = load i64, i64 addrspace(1) * %in
  %b = load i64, i64 addrspace(1) * %b_ptr
  %result = shl i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;EG: {{^}}shl_v2i64:
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
;EG-DAG: LSHR {{\*? *}}[[COMPSHA]]
;EG-DAG: LSHR {{\*? *}}[[COMPSHB]]
;EG-DAG: LSHR {{.*}}, 1
;EG-DAG: LSHR {{.*}}, 1
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: LSHL {{.*}}, [[SHA]]
;EG-DAG: LSHL {{.*}}, [[SHB]]
;EG-DAG: LSHL {{.*}}, [[SHA]]
;EG-DAG: LSHL {{.*}}, [[SHB]]
;EG-DAG: LSHL
;EG-DAG: LSHL
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
;EG-DAG: CNDE_INT {{.*}}, 0.0
;EG-DAG: CNDE_INT {{.*}}, 0.0
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT

;SI: {{^}}shl_v2i64:
;SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI: {{^}}shl_v2i64:
;VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @shl_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %in, i64 1
  %a = load <2 x i64>, <2 x i64> addrspace(1) * %in
  %b = load <2 x i64>, <2 x i64> addrspace(1) * %b_ptr
  %result = shl <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

;EG: {{^}}shl_v4i64:
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHC:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHC:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHD:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHD:T[0-9]+\.[XYZW]]]
;EG-DAG: LSHR {{\*? *}}[[COMPSHA]]
;EG-DAG: LSHR {{\*? *}}[[COMPSHB]]
;EG-DAG: LSHR {{\*? *}}[[COMPSHC]]
;EG-DAG: LSHR {{\*? *}}[[COMPSHD]]
;EG-DAG: LSHR {{.*}}, 1
;EG-DAG: LSHR {{.*}}, 1
;EG-DAG: LSHR {{.*}}, 1
;EG-DAG: LSHR {{.*}}, 1
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHC:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHD:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: LSHL {{.*}}, [[SHA]]
;EG-DAG: LSHL {{.*}}, [[SHB]]
;EG-DAG: LSHL {{.*}}, [[SHC]]
;EG-DAG: LSHL {{.*}}, [[SHD]]
;EG-DAG: LSHL {{.*}}, [[SHA]]
;EG-DAG: LSHL {{.*}}, [[SHB]]
;EG-DAG: LSHL {{.*}}, [[SHC]]
;EG-DAG: LSHL {{.*}}, [[SHD]]
;EG-DAG: LSHL
;EG-DAG: LSHL
;EG-DAG: LSHL
;EG-DAG: LSHL
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHC]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHD]], literal
;EG-DAG: CNDE_INT {{.*}}, 0.0
;EG-DAG: CNDE_INT {{.*}}, 0.0
;EG-DAG: CNDE_INT {{.*}}, 0.0
;EG-DAG: CNDE_INT {{.*}}, 0.0
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT

;SI: {{^}}shl_v4i64:
;SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI: {{^}}shl_v4i64:
;VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @shl_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i64 1
  %a = load <4 x i64>, <4 x i64> addrspace(1) * %in
  %b = load <4 x i64>, <4 x i64> addrspace(1) * %b_ptr
  %result = shl <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}

; Make sure load width gets reduced to i32 load.
; GCN-LABEL: {{^}}s_shl_32_i64:
; GCN-DAG: s_load_dword [[LO_A:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb{{$}}
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], [[LO_A]]
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define void @s_shl_32_i64(i64 addrspace(1)* %out, i64 %a) {
  %result = shl i64 %a, 32
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_shl_32_i64:
; GCN-DAG: buffer_load_dword v[[LO_A:[0-9]+]],
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[LO_A]]{{\]}}
define void @v_shl_32_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.in
  %result = shl i64 %a, 32
  store i64 %result, i64 addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind readnone }
