; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; XUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=redwood < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() #0

; FUNC-LABEL: {{^}}lshr_i32:
; SI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @lshr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = lshr i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v2i32:
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @lshr_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %b_ptr
  %result = lshr <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v4i32:
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @lshr_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %b_ptr
  %result = lshr <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_i64:
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG: SUB_INT {{\*? *}}[[COMPSH:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHIFT:T[0-9]+\.[XYZW]]]
; EG: LSHL {{\* *}}[[TEMP:T[0-9]+\.[XYZW]]], [[OPHI:T[0-9]+\.[XYZW]]], {{[[COMPSH]]|PV.[XYZW]}}
; EG-DAG: ADD_INT {{\*? *}}[[BIGSH:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
; EG-DAG: LSHL {{\*? *}}[[OVERF:T[0-9]+\.[XYZW]]], {{[[TEMP]]|PV.[XYZW]}}, 1
; EG-DAG: LSHR {{\*? *}}[[LOSMTMP:T[0-9]+\.[XYZW]]], [[OPLO:T[0-9]+\.[XYZW]]], [[SHIFT]]
; EG-DAG: OR_INT {{\*? *}}[[LOSM:T[0-9]+\.[XYZW]]], {{[[LOSMTMP]]|PV.[XYZW]|PS}}, {{[[OVERF]]|PV.[XYZW]}}
; EG-DAG: LSHR {{\*? *}}[[HISM:T[0-9]+\.[XYZW]]], [[OPHI]], {{PS|[[SHIFT]]|PV\.[XYZW]}}
; EG-DAG: SETGT_UINT {{\*? *}}[[RESC:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
; EG-DAG: CNDE_INT {{\*? *}}[[RESLO:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]|PS}}
; EG-DAG: LSHR {{\*? *}}[[LOBIG:T[0-9]+\.[XYZW]]], [[OPHI]], [[SHIFT]]
; EG-DAG: CNDE_INT {{\*? *}}[[RESHI:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW], .*}}, 0.0
define amdgpu_kernel void @lshr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 1
  %a = load i64, i64 addrspace(1)* %in
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = lshr i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v2i64:
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHA]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHB]]
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
define amdgpu_kernel void @lshr_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %in, i64 1
  %a = load <2 x i64>, <2 x i64> addrspace(1)* %in
  %b = load <2 x i64>, <2 x i64> addrspace(1)* %b_ptr
  %result = lshr <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v4i64:
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHC:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHC:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHD:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHD:T[0-9]+\.[XYZW]]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHA]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHB]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHC]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHD]]
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHC]]
; EG-DAG: LSHR {{.*}}, [[SHD]]
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHC]]
; EG-DAG: LSHR {{.*}}, [[SHD]]
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHC:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHD:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHC]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHD]], literal
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
define amdgpu_kernel void @lshr_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i64 1
  %a = load <4 x i64>, <4 x i64> addrspace(1)* %in
  %b = load <4 x i64>, <4 x i64> addrspace(1)* %b_ptr
  %result = lshr <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}

; Make sure load width gets reduced to i32 load.
; GCN-LABEL: {{^}}s_lshr_32_i64:
; GCN-DAG: s_load_dword [[HI_A:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x14{{$}}
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], [[HI_A]]
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define amdgpu_kernel void @s_lshr_32_i64(i64 addrspace(1)* %out, [8 x i32], i64 %a) {
  %result = lshr i64 %a, 32
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_lshr_32_i64:
; GCN-DAG: buffer_load_dword v[[HI_A:[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GCN-DAG: v_mov_b32_e32 v[[VHI1:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], v[[VHI1]]{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[HI_A]]:[[VHI]]{{\]}}
define amdgpu_kernel void @v_lshr_32_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.in
  %result = lshr i64 %a, 32
  store i64 %result, i64 addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind readnone }
