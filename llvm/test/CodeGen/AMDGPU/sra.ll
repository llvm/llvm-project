; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() #0

; FUNC-LABEL: {{^}}ashr_v2i32:
; SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @ashr_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %b_ptr
  %result = ashr <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ashr_v4i32:
; SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_ashr_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_ashrrev_i32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @ashr_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %b_ptr
  %result = ashr <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ashr_v2i16:
; FIXME: The ashr operation is uniform, but because its operands come from a
; global load we end up with the vector instructions rather than scalar.
; VI: v_ashrrev_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}), sext(v{{[0-9]+}}) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; VI: v_ashrrev_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}), sext(v{{[0-9]+}}) dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
define amdgpu_kernel void @ashr_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %in, i16 1
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b_ptr
  %result = ashr <2 x i16> %a, %b
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ashr_v4i16:
; FIXME: The ashr operation is uniform, but because its operands come from a
; global load we end up with the vector instructions rather than scalar.
; VI: v_ashrrev_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}), sext(v{{[0-9]+}}) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; VI: v_ashrrev_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}), sext(v{{[0-9]+}}) dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_ashrrev_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}), sext(v{{[0-9]+}}) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; VI: v_ashrrev_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}), sext(v{{[0-9]+}}) dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
define amdgpu_kernel void @ashr_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %in, i16 1
  %a = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %b = load <4 x i16>, <4 x i16> addrspace(1)* %b_ptr
  %result = ashr <4 x i16> %a, %b
  store <4 x i16> %result, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_ashr_i64:
; GCN: s_ashr_i64 s[{{[0-9]}}:{{[0-9]}}], s[{{[0-9]}}:{{[0-9]}}], 8

; EG: ASHR
define amdgpu_kernel void @s_ashr_i64(i64 addrspace(1)* %out, i32 %in) {
entry:
  %in.ext = sext i32 %in to i64
  %ashr = ashr i64 %in.ext, 8
  store i64 %ashr, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ashr_i64_2:
; SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

; VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG: SUB_INT {{\*? *}}[[COMPSH:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHIFT:T[0-9]+\.[XYZW]]]
; EG: LSHL {{\* *}}[[TEMP:T[0-9]+\.[XYZW]]], [[OPHI:T[0-9]+\.[XYZW]]], {{[[COMPSH]]|PV.[XYZW]}}
; EG-DAG: ADD_INT {{\*? *}}[[BIGSH:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
; EG-DAG: LSHL {{\*? *}}[[OVERF:T[0-9]+\.[XYZW]]], {{[[TEMP]]|PV.[XYZW]}}, 1
; EG-DAG: LSHR {{\*? *}}[[LOSMTMP:T[0-9]+\.[XYZW]]], [[OPLO:T[0-9]+\.[XYZW]]], [[SHIFT]]
; EG-DAG: OR_INT {{\*? *}}[[LOSM:T[0-9]+\.[XYZW]]], {{[[LOSMTMP]]|PV.[XYZW]|PS}}, {{[[OVERF]]|PV.[XYZW]}}
; EG-DAG: ASHR {{\*? *}}[[HISM:T[0-9]+\.[XYZW]]], [[OPHI]], {{PS|PV.[XYZW]|[[SHIFT]]}}
; EG-DAG: ASHR {{\*? *}}[[LOBIG:T[0-9]+\.[XYZW]]], [[OPHI]], literal
; EG-DAG: ASHR {{\*? *}}[[HIBIG:T[0-9]+\.[XYZW]]], [[OPHI]], literal
; EG-DAG: SETGT_UINT {{\*? *}}[[RESC:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
; EG-DAG: CNDE_INT {{\*? *}}[[RESLO:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}
; EG-DAG: CNDE_INT {{\*? *}}[[RESHI:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @ashr_i64_2(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
entry:
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 1
  %a = load i64, i64 addrspace(1)* %in
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = ashr i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ashr_v2i64:
; SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHA]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHB]]
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: ASHR {{.*}}, [[SHA]]
; EG-DAG: ASHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: ASHR {{.*}}, literal
; EG-DAG: ASHR {{.*}}, literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
define amdgpu_kernel void @ashr_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %in, i64 1
  %a = load <2 x i64>, <2 x i64> addrspace(1)* %in
  %b = load <2 x i64>, <2 x i64> addrspace(1)* %b_ptr
  %result = ashr <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

; FIXME: Broken on r600
; XFUNC-LABEL: {{^}}s_ashr_v2i64:
; XGCN: s_ashr_i64 {{s\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], s[0-9]+}}
; XGCN: s_ashr_i64 {{s\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], s[0-9]+}}
; define amdgpu_kernel void @s_ashr_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in, <2 x i64> %a, <2 x i64> %b) {
;   %result = ashr <2 x i64> %a, %b
;   store <2 x i64> %result, <2 x i64> addrspace(1)* %out
;   ret void
; }

; FUNC-LABEL: {{^}}ashr_v4i64:
; SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_ashr_i64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

; VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_ashrrev_i64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

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
; EG-DAG: ASHR {{.*}}, [[SHA]]
; EG-DAG: ASHR {{.*}}, [[SHB]]
; EG-DAG: ASHR {{.*}}, [[SHC]]
; EG-DAG: ASHR {{.*}}, [[SHD]]
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHC:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHD:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: ASHR {{.*}}, literal
; EG-DAG: ASHR {{.*}}, literal
; EG-DAG: ASHR {{.*}}, literal
; EG-DAG: ASHR {{.*}}, literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHC]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHD]], literal
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
define amdgpu_kernel void @ashr_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i64 1
  %a = load <4 x i64>, <4 x i64> addrspace(1)* %in
  %b = load <4 x i64>, <4 x i64> addrspace(1)* %b_ptr
  %result = ashr <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_ashr_32_i64:
; GCN: s_load_dword s[[HI:[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN: s_ashr_i32 s[[SHIFT:[0-9]+]], s[[HI]], 31
; GCN: s_add_u32 s{{[0-9]+}}, s[[HI]], s{{[0-9]+}}
; GCN: s_addc_u32 s{{[0-9]+}}, s[[SHIFT]], s{{[0-9]+}}
define amdgpu_kernel void @s_ashr_32_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %result = ashr i64 %a, 32
  %add = add i64 %result, %b
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_ashr_32_i64:
; SI: buffer_load_dword v[[HI:[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; VI: flat_load_dword v[[HI:[0-9]+]]
; GCN: v_ashrrev_i32_e32 v[[SHIFT:[0-9]+]], 31, v[[HI]]
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v{{\[}}[[HI]]:[[SHIFT]]{{\]}}
define amdgpu_kernel void @v_ashr_32_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.in
  %result = ashr i64 %a, 32
  store i64 %result, i64 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}s_ashr_63_i64:
; GCN: s_load_dword s[[HI:[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN: s_ashr_i32 s[[SHIFT:[0-9]+]], s[[HI]], 31
; GCN: s_add_u32 {{s[0-9]+}}, s[[SHIFT]], {{s[0-9]+}}
; GCN: s_addc_u32 {{s[0-9]+}}, s[[SHIFT]], {{s[0-9]+}}
define amdgpu_kernel void @s_ashr_63_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %result = ashr i64 %a, 63
  %add = add i64 %result, %b
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_ashr_63_i64:
; SI: buffer_load_dword v[[HI:[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; VI: flat_load_dword v[[HI:[0-9]+]]
; GCN: v_ashrrev_i32_e32 v[[SHIFT:[0-9]+]], 31, v[[HI]]
; GCN: v_mov_b32_e32 v[[COPY:[0-9]+]], v[[SHIFT]]
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v{{\[}}[[SHIFT]]:[[COPY]]{{\]}}
define amdgpu_kernel void @v_ashr_63_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.in
  %result = ashr i64 %a, 63
  store i64 %result, i64 addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind readnone }
