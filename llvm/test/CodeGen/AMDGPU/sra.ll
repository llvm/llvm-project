;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s
;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=VI %s

;EG-LABEL: {{^}}ashr_v2i32:
;EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-LABEL: {{^}}ashr_v2i32:
;SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

;VI-LABEL: {{^}}ashr_v2i32:
;VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @ashr_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1) * %in
  %b = load <2 x i32>, <2 x i32> addrspace(1) * %b_ptr
  %result = ashr <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG-LABEL: {{^}}ashr_v4i32:
;EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-LABEL: {{^}}ashr_v4i32:
;SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

;VI-LABEL: {{^}}ashr_v4i32:
;VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @ashr_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1) * %in
  %b = load <4 x i32>, <4 x i32> addrspace(1) * %b_ptr
  %result = ashr <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

;EG-LABEL: {{^}}ashr_i64:
;EG: ASHR

;SI-LABEL: {{^}}ashr_i64:
;SI: s_ashr_i64 s[{{[0-9]}}:{{[0-9]}}], s[{{[0-9]}}:{{[0-9]}}], 8

;VI-LABEL: {{^}}ashr_i64:
;VI: s_ashr_i64 s[{{[0-9]}}:{{[0-9]}}], s[{{[0-9]}}:{{[0-9]}}], 8

define void @ashr_i64(i64 addrspace(1)* %out, i32 %in) {
entry:
  %0 = sext i32 %in to i64
  %1 = ashr i64 %0, 8
  store i64 %1, i64 addrspace(1)* %out
  ret void
}

;EG-LABEL: {{^}}ashr_i64_2:
;EG: SUB_INT {{\*? *}}[[COMPSH:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHIFT:T[0-9]+\.[XYZW]]]
;EG: LSHL {{\* *}}[[TEMP:T[0-9]+\.[XYZW]]], [[OPHI:T[0-9]+\.[XYZW]]], {{[[COMPSH]]|PV.[XYZW]}}
;EG: LSHL {{\*? *}}[[OVERF:T[0-9]+\.[XYZW]]], {{[[TEMP]]|PV.[XYZW]}}, 1
;EG_CHECK-DAG: ADD_INT {{\*? *}}[[BIGSH:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-DAG: LSHR {{\*? *}}[[LOSMTMP:T[0-9]+\.[XYZW]]], [[OPLO:T[0-9]+\.[XYZW]]], [[SHIFT]]
;EG-DAG: OR_INT {{\*? *}}[[LOSM:T[0-9]+\.[XYZW]]], {{[[LOSMTMP]]|PV.[XYZW]}}, {{[[OVERF]]|PV.[XYZW]}}
;EG-DAG: ASHR {{\*? *}}[[HISM:T[0-9]+\.[XYZW]]], [[OPHI]], {{PS|[[SHIFT]]}}
;EG-DAG: ASHR {{\*? *}}[[LOBIG:T[0-9]+\.[XYZW]]], [[OPHI]], literal
;EG-DAG: ASHR {{\*? *}}[[HIBIG:T[0-9]+\.[XYZW]]], [[OPHI]], literal
;EG-DAG: SETGT_UINT {{\*? *}}[[RESC:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
;EG-DAG: CNDE_INT {{\*? *}}[[RESLO:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}
;EG-DAG: CNDE_INT {{\*? *}}[[RESHI:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}

;SI-LABEL: {{^}}ashr_i64_2:
;SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI-LABEL: {{^}}ashr_i64_2:
;VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @ashr_i64_2(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
entry:
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 1
  %a = load i64, i64 addrspace(1) * %in
  %b = load i64, i64 addrspace(1) * %b_ptr
  %result = ashr i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;EG-LABEL: {{^}}ashr_v2i64:
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
;EG-DAG: LSHL {{\*? *}}[[COMPSHA]]
;EG-DAG: LSHL {{\*? *}}[[COMPSHB]]
;EG-DAG: LSHL {{.*}}, 1
;EG-DAG: LSHL {{.*}}, 1
;EG-DAG: ASHR {{.*}}, [[SHA]]
;EG-DAG: ASHR {{.*}}, [[SHB]]
;EG-DAG: LSHR {{.*}}, [[SHA]]
;EG-DAG: LSHR {{.*}}, [[SHB]]
;EG-DAG: OR_INT
;EG-DAG: OR_INT
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ASHR
;EG-DAG: ASHR
;EG-DAG: ASHR {{.*}}, literal
;EG-DAG: ASHR {{.*}}, literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT

;SI-LABEL: {{^}}ashr_v2i64:
;SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI-LABEL: {{^}}ashr_v2i64:
;VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @ashr_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %in, i64 1
  %a = load <2 x i64>, <2 x i64> addrspace(1) * %in
  %b = load <2 x i64>, <2 x i64> addrspace(1) * %b_ptr
  %result = ashr <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

;EG-LABEL: {{^}}ashr_v4i64:
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHC:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHC:T[0-9]+\.[XYZW]]]
;EG-DAG: SUB_INT {{\*? *}}[[COMPSHD:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHD:T[0-9]+\.[XYZW]]]
;EG-DAG: LSHL {{\*? *}}[[COMPSHA]]
;EG-DAG: LSHL {{\*? *}}[[COMPSHB]]
;EG-DAG: LSHL {{\*? *}}[[COMPSHC]]
;EG-DAG: LSHL {{\*? *}}[[COMPSHD]]
;EG-DAG: LSHL {{.*}}, 1
;EG-DAG: LSHL {{.*}}, 1
;EG-DAG: LSHL {{.*}}, 1
;EG-DAG: LSHL {{.*}}, 1
;EG-DAG: ASHR {{.*}}, [[SHA]]
;EG-DAG: ASHR {{.*}}, [[SHB]]
;EG-DAG: ASHR {{.*}}, [[SHC]]
;EG-DAG: ASHR {{.*}}, [[SHD]]
;EG-DAG: LSHR {{.*}}, [[SHA]]
;EG-DAG: LSHR {{.*}}, [[SHB]]
;EG-DAG: LSHR {{.*}}, [[SHA]]
;EG-DAG: LSHR {{.*}}, [[SHB]]
;EG-DAG: OR_INT
;EG-DAG: OR_INT
;EG-DAG: OR_INT
;EG-DAG: OR_INT
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHC:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ADD_INT  {{\*? *}}[[BIGSHD:T[0-9]+\.[XYZW]]]{{.*}}, literal
;EG-DAG: ASHR
;EG-DAG: ASHR
;EG-DAG: ASHR
;EG-DAG: ASHR
;EG-DAG: ASHR {{.*}}, literal
;EG-DAG: ASHR {{.*}}, literal
;EG-DAG: ASHR {{.*}}, literal
;EG-DAG: ASHR {{.*}}, literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHC]], literal
;EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHD]], literal
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT
;EG-DAG: CNDE_INT

;SI-LABEL: {{^}}ashr_v4i64:
;SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
;SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

;VI-LABEL: {{^}}ashr_v4i64:
;VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
;VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

define void @ashr_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i64 1
  %a = load <4 x i64>, <4 x i64> addrspace(1) * %in
  %b = load <4 x i64>, <4 x i64> addrspace(1) * %b_ptr
  %result = ashr <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}

