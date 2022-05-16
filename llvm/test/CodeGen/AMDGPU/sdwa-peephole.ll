; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -amdgpu-sdwa-peephole=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=NOSDWA,GCN %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -amdgpu-sdwa-peephole -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI,GFX89,SDWA,GCN %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -amdgpu-sdwa-peephole -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX9,GFX9_10,SDWA,GCN %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx1010 -amdgpu-sdwa-peephole -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX10,GFX9_10,SDWA,GCN %s

; GCN-LABEL: {{^}}add_shr_i32:
; NOSDWA: v_lshrrev_b32_e32 v[[DST:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_add_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v[[DST]]
; NOSDWA-NOT: v_add_{{(_co)?}}_u32_sdwa

; VI: v_add_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; GFX9: v_add_u32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; GFX10: v_add_nc_u32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

define amdgpu_kernel void @add_shr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %a = load i32, i32 addrspace(1)* %in, align 4
  %shr = lshr i32 %a, 16
  %add = add i32 %a, %shr
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sub_shr_i32:
; NOSDWA: v_lshrrev_b32_e32 v[[DST:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_subrev_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v[[DST]]
; NOSDWA-NOT: v_subrev_{{(_co)?}}_u32_sdwa

; VI: v_subrev_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; GFX9: v_sub_u32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX10: v_sub_nc_u32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
define amdgpu_kernel void @sub_shr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %a = load i32, i32 addrspace(1)* %in, align 4
  %shr = lshr i32 %a, 16
  %sub = sub i32 %shr, %a
  store i32 %sub, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_shr_i32:
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v[[DST0]], v[[DST1]]
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA: v_mul_u32_u24_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1

define amdgpu_kernel void @mul_shr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in1, i32 addrspace(1)* %in2) #0 {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gep1 = getelementptr i32, i32 addrspace(1)* %in1, i32 %idx
  %gep2 = getelementptr i32, i32 addrspace(1)* %in2, i32 %idx
  %a = load i32, i32 addrspace(1)* %gep1, align 4
  %b = load i32, i32 addrspace(1)* %gep2, align 4
  %shra = lshr i32 %a, 16
  %shrb = lshr i32 %b, 16
  %mul = mul i32 %shra, %shrb
  store i32 %mul, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_i16:
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa
; GFX89: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX10: v_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-NOT: v_mul_u32_u24_sdwa

define amdgpu_kernel void @mul_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %ina, i16 addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr i16, i16 addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr i16, i16 addrspace(1)* %inb, i32 %idx
  %a = load i16, i16 addrspace(1)* %gepa, align 4
  %b = load i16, i16 addrspace(1)* %gepb, align 4
  %mul = mul i16 %a, %b
  store i16 %mul, i16 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v2i16:
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mul_lo_u16_e32 v[[DST_MUL:[0-9]+]], v[[DST1]], v[[DST0]]
; NOSDWA: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MUL]]
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[DST_SHL]]
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; VI-DAG: v_mul_lo_u16_e32 v[[DST_MUL_LO:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v[[DST_MUL_HI:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL_LO]], v[[DST_MUL_HI]]

; GFX9_10: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %ina, <2 x i16> addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %inb, i32 %idx
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gepa, align 4
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %gepb, align 4
  %mul = mul <2 x i16> %a, %b
  store <2 x i16> %mul, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v4i16:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; VI-DAG: v_mul_lo_u16_e32 v[[DST_MUL0:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v[[DST_MUL1:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_lo_u16_e32 v[[DST_MUL2:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v[[DST_MUL3:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL2]], v[[DST_MUL3]]
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL0]], v[[DST_MUL1]]

; GFX9_10-DAG: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %ina, <4 x i16> addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %inb, i32 %idx
  %a = load <4 x i16>, <4 x i16> addrspace(1)* %gepa, align 4
  %b = load <4 x i16>, <4 x i16> addrspace(1)* %gepb, align 4
  %mul = mul <4 x i16> %a, %b
  store <4 x i16> %mul, <4 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v8i16:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; VI-DAG: v_mul_lo_u16_e32 v[[DST_MUL0:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v[[DST_MUL1:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_lo_u16_e32 v[[DST_MUL2:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v[[DST_MUL3:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_lo_u16_e32 v[[DST_MUL4:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v[[DST_MUL5:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_lo_u16_e32 v[[DST_MUL6:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v[[DST_MUL7:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL6]], v[[DST_MUL7]]
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL4]], v[[DST_MUL5]]
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL2]], v[[DST_MUL3]]
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL0]], v[[DST_MUL1]]

; GFX9_10-DAG: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> addrspace(1)* %ina, <8 x i16> addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr <8 x i16>, <8 x i16> addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr <8 x i16>, <8 x i16> addrspace(1)* %inb, i32 %idx
  %a = load <8 x i16>, <8 x i16> addrspace(1)* %gepa, align 4
  %b = load <8 x i16>, <8 x i16> addrspace(1)* %gepb, align 4
  %mul = mul <8 x i16> %a, %b
  store <8 x i16> %mul, <8 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_half:
; NOSDWA: v_mul_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_f16_sdwa
; SDWA: v_mul_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-NOT: v_mul_f16_sdwa

define amdgpu_kernel void @mul_half(half addrspace(1)* %out, half addrspace(1)* %ina, half addrspace(1)* %inb) #0 {
entry:
  %a = load half, half addrspace(1)* %ina, align 4
  %b = load half, half addrspace(1)* %inb, align 4
  %mul = fmul half %a, %b
  store half %mul, half addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v2half:
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mul_f16_e32 v[[DST_MUL:[0-9]+]], v[[DST0]], v[[DST1]]
; NOSDWA: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MUL]]
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[DST_SHL]]
; NOSDWA-NOT: v_mul_f16_sdwa

; VI-DAG: v_mul_f16_sdwa v[[DST_MUL_HI:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_f16_e32 v[[DST_MUL_LO:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL_LO]], v[[DST_MUL_HI]]

; GFX9_10: v_pk_mul_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v2half(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %ina, <2 x half> addrspace(1)* %inb) #0 {
entry:
  %a = load <2 x half>, <2 x half> addrspace(1)* %ina, align 4
  %b = load <2 x half>, <2 x half> addrspace(1)* %inb, align 4
  %mul = fmul <2 x half> %a, %b
  store <2 x half> %mul, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v4half:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_f16_sdwa

; VI-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; GFX9_10-DAG: v_pk_mul_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v4half(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %ina, <4 x half> addrspace(1)* %inb) #0 {
entry:
  %a = load <4 x half>, <4 x half> addrspace(1)* %ina, align 4
  %b = load <4 x half>, <4 x half> addrspace(1)* %inb, align 4
  %mul = fmul <4 x half> %a, %b
  store <4 x half> %mul, <4 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v8half:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_f16_sdwa

; VI-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; GFX9_10-DAG: v_pk_mul_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10-DAG: v_pk_mul_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v8half(<8 x half> addrspace(1)* %out, <8 x half> addrspace(1)* %ina, <8 x half> addrspace(1)* %inb) #0 {
entry:
  %a = load <8 x half>, <8 x half> addrspace(1)* %ina, align 4
  %b = load <8 x half>, <8 x half> addrspace(1)* %inb, align 4
  %mul = fmul <8 x half> %a, %b
  store <8 x half> %mul, <8 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_i8:
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa
; GFX89: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX10: v_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-NOT: v_mul_u32_u24_sdwa

define amdgpu_kernel void @mul_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %ina, i8 addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr i8, i8 addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr i8, i8 addrspace(1)* %inb, i32 %idx
  %a = load i8, i8 addrspace(1)* %gepa, align 4
  %b = load i8, i8 addrspace(1)* %gepb, align 4
  %mul = mul i8 %a, %b
  store i8 %mul, i8 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v2i8:
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; VI: v_mul_lo_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1

; GFX9-DAG: v_mul_lo_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1
; GFX9-DAG: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16

; GFX9: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD

; GFX10: v_lshlrev_b16 v{{[0-9]+}}, 8, v
; GFX10: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
define amdgpu_kernel void @mul_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(1)* %ina, <2 x i8> addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr <2 x i8>, <2 x i8> addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr <2 x i8>, <2 x i8> addrspace(1)* %inb, i32 %idx
  %a = load <2 x i8>, <2 x i8> addrspace(1)* %gepa, align 4
  %b = load <2 x i8>, <2 x i8> addrspace(1)* %gepb, align 4
  %mul = mul <2 x i8> %a, %b
  store <2 x i8> %mul, <2 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v4i8:
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; VI-DAG: v_mul_lo_u16_sdwa
; VI-DAG: v_mul_lo_u16_sdwa
; VI-DAG: v_mul_lo_u16_sdwa

; GFX9-DAG: v_mul_lo_u16_sdwa
; GFX9-DAG: v_mul_lo_u16_sdwa
; GFX9-DAG: v_mul_lo_u16_sdwa

; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16

define amdgpu_kernel void @mul_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %ina, <4 x i8> addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %inb, i32 %idx
  %a = load <4 x i8>, <4 x i8> addrspace(1)* %gepa, align 4
  %b = load <4 x i8>, <4 x i8> addrspace(1)* %gepb, align 4
  %mul = mul <4 x i8> %a, %b
  store <4 x i8> %mul, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v8i8:
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; VI-DAG: v_mul_lo_u16_sdwa
; VI-DAG: v_mul_lo_u16_sdwa
; VI-DAG: v_mul_lo_u16_sdwa
; VI-DAG: v_mul_lo_u16_sdwa
; VI-DAG: v_mul_lo_u16_sdwa
; VI-DAG: v_mul_lo_u16_sdwa

; GFX9-DAG: v_mul_lo_u16_sdwa
; GFX9-DAG: v_mul_lo_u16_sdwa
; GFX9-DAG: v_mul_lo_u16_sdwa
; GFX9-DAG: v_mul_lo_u16_sdwa
; GFX9-DAG: v_mul_lo_u16_sdwa
; GFX9-DAG: v_mul_lo_u16_sdwa

; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16
; GFX10-DAG: v_mul_lo_u16

define amdgpu_kernel void @mul_v8i8(<8 x i8> addrspace(1)* %out, <8 x i8> addrspace(1)* %ina, <8 x i8> addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr <8 x i8>, <8 x i8> addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr <8 x i8>, <8 x i8> addrspace(1)* %inb, i32 %idx
  %a = load <8 x i8>, <8 x i8> addrspace(1)* %gepa, align 4
  %b = load <8 x i8>, <8 x i8> addrspace(1)* %gepb, align 4
  %mul = mul <8 x i8> %a, %b
  store <8 x i8> %mul, <8 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sitofp_v2i16_to_v2f16:
; NOSDWA-DAG: v_cvt_f16_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-DAG: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA-DAG: v_cvt_f16_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_cvt_f16_i16_sdwa

; SDWA-DAG: v_cvt_f16_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-DAG: v_cvt_f16_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}} dst_sel:{{(WORD_1|DWORD)?}} dst_unused:UNUSED_PAD src0_sel:WORD_1

; FIXME: Should be able to avoid or
define amdgpu_kernel void @sitofp_v2i16_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a) #0 {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %r.val = sitofp <2 x i16> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}


; GCN-LABEL: {{^}}mac_v2half:
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mac_f16_e32 v[[DST_MAC:[0-9]+]], v[[DST0]], v[[DST1]]
; NOSDWA: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MAC]]
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[DST_SHL]]
; NOSDWA-NOT: v_mac_f16_sdwa

; VI: v_mac_f16_sdwa v[[DST_MAC:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MAC]]

; GFX9_10: v_pk_mul_f16 v[[DST_MUL:[0-9]+]], v{{[0-9]+}}, v[[SRC:[0-9]+]]
; GFX9_10: v_pk_add_f16 v{{[0-9]+}}, v[[DST_MUL]], v[[SRC]]

define amdgpu_kernel void @mac_v2half(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %ina, <2 x half> addrspace(1)* %inb) #0 {
entry:
  %a = load <2 x half>, <2 x half> addrspace(1)* %ina, align 4
  %b = load <2 x half>, <2 x half> addrspace(1)* %inb, align 4
  %mul = fmul <2 x half> %a, %b
  %mac = fadd <2 x half> %mul, %b
  store <2 x half> %mac, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}immediate_mul_v2i16:
; NOSDWA-NOT: v_mul_u32_u24_sdwa
; VI-DAG: v_mov_b32_e32 v[[M321:[0-9]+]], 0x141
; VI-DAG: v_mul_lo_u16_e32 v{{[0-9]+}}, 0x7b, v{{[0-9]+}}
; VI-DAG: v_mul_lo_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v[[M321]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD

; GFX9: s_mov_b32 s[[IMM:[0-9]+]], 0x141007b
; GFX9: v_pk_mul_lo_u16 v{{[0-9]+}}, v{{[0-9]+}}, s[[IMM]]

; GFX10: v_pk_mul_lo_u16 v{{[0-9]+}}, 0x141007b, v{{[0-9]+}}

define amdgpu_kernel void @immediate_mul_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %in, i32 %idx
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gep, align 4
  %mul = mul <2 x i16> %a, <i16 123, i16 321>
  store <2 x i16> %mul, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; Double use of same src - should not convert it
; GCN-LABEL: {{^}}mulmul_v2i16:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_lo_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; VI: v_mul_lo_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD

; GFX9_10: v_pk_mul_lo_u16 v[[DST1:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GFX9_10: v_pk_mul_lo_u16 v{{[0-9]+}}, v[[DST1]], v{{[0-9]+}}

define amdgpu_kernel void @mulmul_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %ina, <2 x i16> addrspace(1)* %inb) #0 {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gepa = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %ina, i32 %idx
  %gepb = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %inb, i32 %idx
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gepa, align 4
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %gepb, align 4
  %mul = mul <2 x i16> %a, %b
  %mul2 = mul <2 x i16> %mul, %b
  store <2 x i16> %mul2, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}add_bb_v2i16:
; NOSDWA-NOT: v_add_{{(_co)?}}_u32_sdwa

; VI: v_add_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1

; GFX9_10: v_pk_add_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @add_bb_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %ina, <2 x i16> addrspace(1)* %inb) #0 {
entry:
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %ina, align 4
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %inb, align 4
  br label %add_label
add_label:
  %add = add <2 x i16> %a, %b
  br label %store_label
store_label:
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out, align 4
  ret void
}


; Check that "pulling out" SDWA operands works correctly.
; GCN-LABEL: {{^}}pulled_out_test:
; NOSDWA-DAG: v_and_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-DAG: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA-DAG: v_and_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-DAG: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_and_b32_sdwa
; NOSDWA-NOT: v_or_b32_sdwa

; VI-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX9-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX10-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX89-DAG: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
;
; GFX10-DAG: v_lshrrev_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
;
; VI-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX9-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX10-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX89-DAG: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
;
; GFX10-DAG: v_lshrrev_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
;
; GFX89: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
;
; GFX10: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; GFX10: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

define amdgpu_kernel void @pulled_out_test(<8 x i8> addrspace(1)* %sourceA, <8 x i8> addrspace(1)* %destValues) #0 {
entry:
  %idxprom = ashr exact i64 15, 32
  %arrayidx = getelementptr inbounds <8 x i8>, <8 x i8> addrspace(1)* %sourceA, i64 %idxprom
  %tmp = load <8 x i8>, <8 x i8> addrspace(1)* %arrayidx, align 8

  %tmp1 = extractelement <8 x i8> %tmp, i32 0
  %tmp2 = extractelement <8 x i8> %tmp, i32 1
  %tmp3 = extractelement <8 x i8> %tmp, i32 2
  %tmp4 = extractelement <8 x i8> %tmp, i32 3
  %tmp5 = extractelement <8 x i8> %tmp, i32 4
  %tmp6 = extractelement <8 x i8> %tmp, i32 5
  %tmp7 = extractelement <8 x i8> %tmp, i32 6
  %tmp8 = extractelement <8 x i8> %tmp, i32 7

  %tmp9 = insertelement <2 x i8> undef, i8 %tmp1, i32 0
  %tmp10 = insertelement <2 x i8> %tmp9, i8 %tmp2, i32 1
  %tmp11 = insertelement <2 x i8> undef, i8 %tmp3, i32 0
  %tmp12 = insertelement <2 x i8> %tmp11, i8 %tmp4, i32 1
  %tmp13 = insertelement <2 x i8> undef, i8 %tmp5, i32 0
  %tmp14 = insertelement <2 x i8> %tmp13, i8 %tmp6, i32 1
  %tmp15 = insertelement <2 x i8> undef, i8 %tmp7, i32 0
  %tmp16 = insertelement <2 x i8> %tmp15, i8 %tmp8, i32 1

  %tmp17 = shufflevector <2 x i8> %tmp10, <2 x i8> %tmp12, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp18 = shufflevector <2 x i8> %tmp14, <2 x i8> %tmp16, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp19 = shufflevector <4 x i8> %tmp17, <4 x i8> %tmp18, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>

  %arrayidx5 = getelementptr inbounds <8 x i8>, <8 x i8> addrspace(1)* %destValues, i64 %idxprom
  store <8 x i8> %tmp19, <8 x i8> addrspace(1)* %arrayidx5, align 8
  ret void
}

; GCN-LABEL: {{^}}sdwa_crash_inlineasm_def:
; GCN: s_mov_b32 s{{[0-9]+}}, 0xffff
; GCN: v_and_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
;
; TODO: Why is the constant not peepholed into the v_or_b32_e32?
;
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, 0x10000,
; SDWA: v_or_b32_e32 v{{[0-9]+}}, 0x10000,
define amdgpu_kernel void @sdwa_crash_inlineasm_def() #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb11, %bb
  %tmp = phi <2 x i32> [ %tmp12, %bb11 ], [ undef, %bb ]
  br i1 true, label %bb2, label %bb11

bb2:                                              ; preds = %bb1
  %tmp3 = call i32 asm "v_and_b32_e32 $0, $1, $2", "=v,s,v"(i32 65535, i32 undef) #1
  %tmp5 = or i32 %tmp3, 65536
  %tmp6 = insertelement <2 x i32> %tmp, i32 %tmp5, i64 0
  br label %bb11

bb11:                                             ; preds = %bb10, %bb2
  %tmp12 = phi <2 x i32> [ %tmp6, %bb2 ], [ %tmp, %bb1 ]
  store volatile <2 x i32> %tmp12, <2 x i32> addrspace(1)* undef
  br label %bb1
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { "denormal-fp-math"="preserve-sign,preserve-sign" }
