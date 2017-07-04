; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; XUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() #0

declare i32 @llvm.r600.read.tgid.x() #0


;EG: {{^}}shl_v2i32:
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: {{^}}shl_v2i32:
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

;VI: {{^}}shl_v2i32:
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define amdgpu_kernel void @shl_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %b_ptr
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

define amdgpu_kernel void @shl_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %b_ptr
  %result = shl <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_i16:
; SI: v_lshlrev_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @shl_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %b_ptr = getelementptr i16, i16 addrspace(1)* %in, i16 1
  %a = load i16, i16 addrspace(1)* %in
  %b = load i16, i16 addrspace(1)* %b_ptr
  %result = shl i16 %a, %b
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_i16_v_s:
; VI: v_lshlrev_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}

; VI: v_lshlrev_b16_e64 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @shl_i16_v_s(i16 addrspace(1)* %out, i16 addrspace(1)* %in, i16 %b) {
  %a = load i16, i16 addrspace(1)* %in
  %result = shl i16 %a, %b
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_i16_v_compute_s:
; SI: v_lshlrev_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}

; VI: v_lshlrev_b16_e64 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @shl_i16_v_compute_s(i16 addrspace(1)* %out, i16 addrspace(1)* %in, i16 %b) {
  %a = load i16, i16 addrspace(1)* %in
  %b.add = add i16 %b, 3
  %result = shl i16 %a, %b.add
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_i16_computed_amount:
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], 3, v{{[0-9]+}}
; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, [[ADD]], v{{[0-9]+}}
define amdgpu_kernel void @shl_i16_computed_amount(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %b_ptr = getelementptr i16, i16 addrspace(1)* %gep, i16 1
  %a = load volatile i16, i16 addrspace(1)* %in
  %b = load volatile i16, i16 addrspace(1)* %b_ptr
  %b.add = add i16 %b, 3
  %result = shl i16 %a, %b.add
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_i16_i_s:
; GCN: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 12
define amdgpu_kernel void @shl_i16_i_s(i16 addrspace(1)* %out, i16 zeroext %a) {
  %result = shl i16 %a, 12
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_v2i16:
; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @shl_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %b_ptr = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %gep, i16 1
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b_ptr
  %result = shl <2 x i16> %a, %b
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_v4i16:
; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @shl_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %out, i32 %tid
  %b_ptr = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %gep, i16 1
  %a = load <4 x i16>, <4 x i16> addrspace(1)* %gep
  %b = load <4 x i16>, <4 x i16> addrspace(1)* %b_ptr
  %result = shl <4 x i16> %a, %b
  store <4 x i16> %result, <4 x i16> addrspace(1)* %gep.out
  ret void
}

;EG-LABEL: {{^}}shl_i64:
;EG: SUB_INT {{\*? *}}[[COMPSH:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHIFT:T[0-9]+\.[XYZW]]]
;EG: LSHR {{\* *}}[[TEMP:T[0-9]+\.[XYZW]]], [[OPLO:T[0-9]+\.[XYZW]]], {{[[COMPSH]]|PV.[XYZW]}}
;EG-DAG: ADD_INT {{\*? *}}[[BIGSH:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-DAG: LSHR {{\*? *}}[[OVERF:T[0-9]+\.[XYZW]]], {{[[TEMP]]|PV.[XYZW]}}, 1
;EG-DAG: LSHL {{\*? *}}[[HISMTMP:T[0-9]+\.[XYZW]]], [[OPHI:T[0-9]+\.[XYZW]]], [[SHIFT]]
;EG-DAG: OR_INT {{\*? *}}[[HISM:T[0-9]+\.[XYZW]]], {{[[HISMTMP]]|PV.[XYZW]|PS}}, {{[[OVERF]]|PV.[XYZW]}}
;EG-DAG: LSHL {{\*? *}}[[LOSM:T[0-9]+\.[XYZW]]], [[OPLO]], {{PS|[[SHIFT]]|PV.[XYZW]}}
;EG-DAG: SETGT_UINT {{\*? *}}[[RESC:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-DAG: CNDE_INT {{\*? *}}[[RESLO:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}
;EG-DAG: CNDE_INT {{\*? *}}[[RESHI:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW], .*}}, 0.0

; GCN-LABEL: {{^}}shl_i64:
; SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; VI: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @shl_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 1
  %a = load i64, i64 addrspace(1)* %in
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = shl i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;EG-LABEL: {{^}}shl_v2i64:
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

define amdgpu_kernel void @shl_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %in, i64 1
  %a = load <2 x i64>, <2 x i64> addrspace(1)* %in
  %b = load <2 x i64>, <2 x i64> addrspace(1)* %b_ptr
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

define amdgpu_kernel void @shl_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i64 1
  %a = load <4 x i64>, <4 x i64> addrspace(1)* %in
  %b = load <4 x i64>, <4 x i64> addrspace(1)* %b_ptr
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
define amdgpu_kernel void @s_shl_32_i64(i64 addrspace(1)* %out, i64 %a) {
  %result = shl i64 %a, 32
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_shl_32_i64:
; GCN-DAG: buffer_load_dword v[[LO_A:[0-9]+]],
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[LO_A]]{{\]}}
define amdgpu_kernel void @v_shl_32_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tgid.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.in
  %result = shl i64 %a, 32
  store i64 %result, i64 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}s_shl_constant_i64
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_constant_i64(i64 addrspace(1)* %out, i64 %a) {
  %shl = shl i64 281474976710655, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_shl_constant_i64:
; SI-DAG: buffer_load_dword [[VAL:v[0-9]+]]
; SI-DAG: s_mov_b32 s[[KLO:[0-9]+]], 0xab19b207
; SI-DAG: s_movk_i32 s[[KHI:[0-9]+]], 0x11e{{$}}
; SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\]}}, s{{\[}}[[KLO]]:[[KHI]]{{\]}}, [[VAL]]
; SI: buffer_store_dwordx2
define amdgpu_kernel void @v_shl_constant_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load i64, i64 addrspace(1)* %aptr, align 8
  %shl = shl i64 1231231234567, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_shl_i64_32_bit_constant:
; SI-DAG: buffer_load_dword [[VAL:v[0-9]+]]
; SI-DAG: s_mov_b32 s[[KLO:[0-9]+]], 0x12d687{{$}}
; SI-DAG: s_mov_b32 s[[KHI:[0-9]+]], 0{{$}}
; SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\]}}, s{{\[}}[[KLO]]:[[KHI]]{{\]}}, [[VAL]]
define amdgpu_kernel void @v_shl_i64_32_bit_constant(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load i64, i64 addrspace(1)* %aptr, align 8
  %shl = shl i64 1234567, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_shl_inline_imm_64_i64:
; SI: v_lshl_b64 {{v\[[0-9]+:[0-9]+\]}}, 64, {{v[0-9]+}}
define amdgpu_kernel void @v_shl_inline_imm_64_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load i64, i64 addrspace(1)* %aptr, align 8
  %shl = shl i64 64, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_64_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, 64, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_64_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 64, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_1_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, 1, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_1_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 1, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_1.0_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, 1.0, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_1.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 4607182418800017408, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_neg_1.0_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, -1.0, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_neg_1.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 13830554455654793216, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_0.5_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, 0.5, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_0.5_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 4602678819172646912, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_neg_0.5_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, -0.5, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_neg_0.5_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 13826050856027422720, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_2.0_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, 2.0, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_2.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 4611686018427387904, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_neg_2.0_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, -2.0, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_neg_2.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 13835058055282163712, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_4.0_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, 4.0, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 4616189618054758400, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_imm_neg_4.0_i64:
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, -4.0, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_neg_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 13839561654909534208, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}


; Test with the 64-bit integer bitpattern for a 32-bit float in the
; low 32-bits, which is not a valid 64-bit inline immmediate.

; FUNC-LABEL: {{^}}s_shl_inline_imm_f32_4.0_i64:
; SI-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 4.0
; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 0{{$}}
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_f32_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 1082130432, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: Copy of -1 register
; FUNC-LABEL: {{^}}s_shl_inline_imm_f32_neg_4.0_i64:
; SI-DAG: s_mov_b32 s[[K_LO:[0-9]+]], -4.0
; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], -1{{$}}
; SI-DAG: s_mov_b32 s[[K_HI_COPY:[0-9]+]], s[[K_HI]]
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI_COPY]]{{\]}}, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_imm_f32_neg_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 -1065353216, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; Shift into upper 32-bits
; FUNC-LABEL: {{^}}s_shl_inline_high_imm_f32_4.0_i64:
; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 4.0
; SI-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 0{{$}}
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_high_imm_f32_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 4647714815446351872, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_shl_inline_high_imm_f32_neg_4.0_i64:
; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], -4.0
; SI-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 0{{$}}
; SI: s_lshl_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}, s{{[0-9]+}}
define amdgpu_kernel void @s_shl_inline_high_imm_f32_neg_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %shl = shl i64 13871086852301127680, %a
  store i64 %shl, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_mul2:
; GCN: s_lshl_b32 s{{[0-9]}}, s{{[0-9]}}, 1
define amdgpu_kernel void @test_mul2(i32 %p) {
   %i = mul i32 %p, 2
   store volatile i32 %i, i32 addrspace(1)* undef
   ret void
}

attributes #0 = { nounwind readnone }
