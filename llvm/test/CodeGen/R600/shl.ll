;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=VI-CHECK %s

;EG-CHECK: {{^}}shl_v2i32:
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: {{^}}shl_v2i32:
;SI-CHECK: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

;VI-CHECK: {{^}}shl_v2i32:
;VI-CHECK: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI-CHECK: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @shl_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = shl <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: {{^}}shl_v4i32:
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: {{^}}shl_v4i32:
;SI-CHECK: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: v_lshl_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

;VI-CHECK: {{^}}shl_v4i32:
;VI-CHECK: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI-CHECK: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI-CHECK: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI-CHECK: v_lshlrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @shl_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = shl <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: {{^}}shl_i64:
;EG-CHECK: SUB_INT {{\*? *}}[[COMPSH:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHIFT:T[0-9]+\.[XYZW]]]
;EG-CHECK: LSHR {{\* *}}[[TEMP:T[0-9]+\.[XYZW]]], [[OPLO:T[0-9]+\.[XYZW]]], {{[[COMPSH]]|PV.[XYZW]}}
;EG-CHECK: LSHR {{\*? *}}[[OVERF:T[0-9]+\.[XYZW]]], {{[[TEMP]]|PV.[XYZW]}}, 1
;EG_CHECK-DAG: ADD_INT {{\*? *}}[[BIGSH:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-CHECK-DAG: LSHL {{\*? *}}[[HISMTMP:T[0-9]+\.[XYZW]]], [[OPHI:T[0-9]+\.[XYZW]]], [[SHIFT]]
;EG-CHECK-DAG: OR_INT {{\*? *}}[[HISM:T[0-9]+\.[XYZW]]], {{[[HISMTMP]]|PV.[XYZW]}}, {{[[OVERF]]|PV.[XYZW]}}
;EG-CHECK-DAG: LSHL {{\*? *}}[[LOSM:T[0-9]+\.[XYZW]]], [[OPLO]], {{PS|[[SHIFT]]}}
;EG-CHECK-DAG: SETGT_UINT {{\*? *}}[[RESC:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-CHECK-DAG: CNDE_INT {{\*? *}}[[RESLO:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}
;EG-CHECK-DAG: CNDE_INT {{\*? *}}[[RESHI:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW], .*}}, 0.0

;SI-CHECK: {{^}}shl_i64:
;SI-CHECK: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI-CHECK: {{^}}shl_i64:
;VI-CHECK: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @shl_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %b_ptr = getelementptr i64 addrspace(1)* %in, i64 1
  %a = load i64 addrspace(1) * %in
  %b = load i64 addrspace(1) * %b_ptr
  %result = shl i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;EG-CHECK: {{^}}shl_v2i64:
;EG-CHECK-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
;EG-CHECK-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
;EG-CHECK-DAG: LSHR {{\*? *}}[[COMPSHA]]
;EG-CHECK-DAG: LSHR {{\*? *}}[[COMPSHB]]
;EG-CHECK-DAG: LSHR {{.*}}, 1
;EG-CHECK-DAG: LSHR {{.*}}, 1
;EG-CHECK-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-CHECK-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-CHECK-DAG: LSHL {{.*}}, [[SHA]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHB]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHA]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHB]]
;EG-CHECK-DAG: LSHL
;EG-CHECK-DAG: LSHL
;EG-CHECK-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
;EG-CHECK-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
;EG-CHECK-DAG: CNDE_INT {{.*}}, 0.0
;EG-CHECK-DAG: CNDE_INT {{.*}}, 0.0
;EG-CHECK-DAG: CNDE_INT
;EG-CHECK-DAG: CNDE_INT

;SI-CHECK: {{^}}shl_v2i64:
;SI-CHECK: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI-CHECK: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI-CHECK: {{^}}shl_v2i64:
;VI-CHECK: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI-CHECK: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @shl_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i64> addrspace(1)* %in, i64 1
  %a = load <2 x i64> addrspace(1) * %in
  %b = load <2 x i64> addrspace(1) * %b_ptr
  %result = shl <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

;EG-CHECK: {{^}}shl_v4i64:
;EG-CHECK-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
;EG-CHECK-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
;EG-CHECK-DAG: SUB_INT {{\*? *}}[[COMPSHC:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHC:T[0-9]+\.[XYZW]]]
;EG-CHECK-DAG: SUB_INT {{\*? *}}[[COMPSHD:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHD:T[0-9]+\.[XYZW]]]
;EG-CHECK-DAG: LSHR {{\*? *}}[[COMPSHA]]
;EG-CHECK-DAG: LSHR {{\*? *}}[[COMPSHB]]
;EG-CHECK-DAG: LSHR {{\*? *}}[[COMPSHC]]
;EG-CHECK-DAG: LSHR {{\*? *}}[[COMPSHD]]
;EG-CHECK-DAG: LSHR {{.*}}, 1
;EG-CHECK-DAG: LSHR {{.*}}, 1
;EG-CHECK-DAG: LSHR {{.*}}, 1
;EG-CHECK-DAG: LSHR {{.*}}, 1
;EG-CHECK-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-CHECK-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-CHECK-DAG: ADD_INT  {{\*? *}}[[BIGSHC:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-CHECK-DAG: ADD_INT  {{\*? *}}[[BIGSHD:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-CHECK-DAG: LSHL {{.*}}, [[SHA]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHB]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHC]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHD]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHA]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHB]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHC]]
;EG-CHECK-DAG: LSHL {{.*}}, [[SHD]]
;EG-CHECK-DAG: LSHL
;EG-CHECK-DAG: LSHL
;EG-CHECK-DAG: LSHL
;EG-CHECK-DAG: LSHL
;EG-CHECK-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
;EG-CHECK-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
;EG-CHECK-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHC]], literal
;EG-CHECK-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHD]], literal
;EG-CHECK-DAG: CNDE_INT {{.*}}, 0.0
;EG-CHECK-DAG: CNDE_INT {{.*}}, 0.0
;EG-CHECK-DAG: CNDE_INT {{.*}}, 0.0
;EG-CHECK-DAG: CNDE_INT {{.*}}, 0.0
;EG-CHECK-DAG: CNDE_INT
;EG-CHECK-DAG: CNDE_INT
;EG-CHECK-DAG: CNDE_INT
;EG-CHECK-DAG: CNDE_INT

;SI-CHECK: {{^}}shl_v4i64:
;SI-CHECK: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI-CHECK: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI-CHECK: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI-CHECK: v_lshl_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI-CHECK: {{^}}shl_v4i64:
;VI-CHECK: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI-CHECK: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI-CHECK: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI-CHECK: v_lshlrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @shl_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i64> addrspace(1)* %in, i64 1
  %a = load <4 x i64> addrspace(1) * %in
  %b = load <4 x i64> addrspace(1) * %b_ptr
  %result = shl <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}
