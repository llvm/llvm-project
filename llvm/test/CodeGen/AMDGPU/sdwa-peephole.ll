; RUN: llc -march=amdgcn -mcpu=fiji -amdgpu-sdwa-peephole=0 -mattr=-fp64-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=NOSDWA -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -amdgpu-sdwa-peephole -mattr=-fp64-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SDWA -check-prefix=GCN %s

; GCN-LABEL: {{^}}add_shr_i32:
; NOSDWA: v_lshrrev_b32_e32 v[[DST:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_add_i32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v[[DST]]
; NOSDWA-NOT: v_add_i32_sdwa

; SDWA: v_add_i32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

define amdgpu_kernel void @add_shr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %a = load i32, i32 addrspace(1)* %in, align 4
  %shr = lshr i32 %a, 16
  %add = add i32 %a, %shr
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sub_shr_i32:
; NOSDWA: v_lshrrev_b32_e32 v[[DST:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_subrev_i32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v[[DST]]
; NOSDWA-NOT: v_subrev_i32_sdwa

; SDWA: v_subrev_i32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

define amdgpu_kernel void @sub_shr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %a = load i32, i32 addrspace(1)* %in, align 4
  %shr = lshr i32 %a, 16
  %sub = sub i32 %shr, %a
  store i32 %sub, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_shr_i32:
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v[[DST1]], v[[DST0]]
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA: v_mul_u32_u24_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1

define amdgpu_kernel void @mul_shr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in1, i32 addrspace(1)* %in2) {
  %a = load i32, i32 addrspace(1)* %in1, align 4
  %b = load i32, i32 addrspace(1)* %in2, align 4
  %shra = lshr i32 %a, 16
  %shrb = lshr i32 %b, 16
  %mul = mul i32 %shra, %shrb
  store i32 %mul, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_i16:
; NOSDWA: v_mul_lo_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa
; SDWA: v_mul_lo_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-NOT: v_mul_u32_u24_sdwa

define amdgpu_kernel void @mul_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %ina, i16 addrspace(1)* %inb) {
entry:
  %a = load i16, i16 addrspace(1)* %ina, align 4
  %b = load i16, i16 addrspace(1)* %inb, align 4
  %mul = mul i16 %a, %b
  store i16 %mul, i16 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v2i16:
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v[[DST_MUL:[0-9]+]], v[[DST1]], v[[DST0]]
; NOSDWA: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MUL]]
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v[[DST_SHL]], v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL_LO:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL_HI:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA: v_or_b32_sdwa v{{[0-9]+}}, v[[DST_MUL_HI]], v[[DST_MUL_LO]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0

define amdgpu_kernel void @mul_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %ina, <2 x i16> addrspace(1)* %inb) {
entry:
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %ina, align 4
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %inb, align 4
  %mul = mul <2 x i16> %a, %b
  store <2 x i16> %mul, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v4i16:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL0:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL1:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL2:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL3:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_or_b32_sdwa v{{[0-9]+}}, v[[DST_MUL3]], v[[DST_MUL2]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; SDWA-DAG: v_or_b32_sdwa v{{[0-9]+}}, v[[DST_MUL1]], v[[DST_MUL0]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0

define amdgpu_kernel void @mul_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %ina, <4 x i16> addrspace(1)* %inb) {
entry:
  %a = load <4 x i16>, <4 x i16> addrspace(1)* %ina, align 4
  %b = load <4 x i16>, <4 x i16> addrspace(1)* %inb, align 4
  %mul = mul <4 x i16> %a, %b
  store <4 x i16> %mul, <4 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v8i16:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL0:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL1:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL2:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL3:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL4:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL5:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL6:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
; SDWA-DAG: v_mul_u32_u24_sdwa v[[DST_MUL7:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_or_b32_sdwa v{{[0-9]+}}, v[[DST_MUL7]], v[[DST_MUL6]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; SDWA-DAG: v_or_b32_sdwa v{{[0-9]+}}, v[[DST_MUL5]], v[[DST_MUL4]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; SDWA-DAG: v_or_b32_sdwa v{{[0-9]+}}, v[[DST_MUL3]], v[[DST_MUL2]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; SDWA-DAG: v_or_b32_sdwa v{{[0-9]+}}, v[[DST_MUL1]], v[[DST_MUL0]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0

define amdgpu_kernel void @mul_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> addrspace(1)* %ina, <8 x i16> addrspace(1)* %inb) {
entry:
  %a = load <8 x i16>, <8 x i16> addrspace(1)* %ina, align 4
  %b = load <8 x i16>, <8 x i16> addrspace(1)* %inb, align 4
  %mul = mul <8 x i16> %a, %b
  store <8 x i16> %mul, <8 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_half:
; NOSDWA: v_mul_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_f16_sdwa
; SDWA: v_mul_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-NOT: v_mul_f16_sdwa

define amdgpu_kernel void @mul_half(half addrspace(1)* %out, half addrspace(1)* %ina, half addrspace(1)* %inb) {
entry:
  %a = load half, half addrspace(1)* %ina, align 4
  %b = load half, half addrspace(1)* %inb, align 4
  %mul = fmul half %a, %b
  store half %mul, half addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v2half:
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mul_f16_e32 v[[DST_MUL:[0-9]+]], v[[DST1]], v[[DST0]]
; NOSDWA: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MUL]]
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v[[DST_SHL]], v{{[0-9]+}}
; NOSDWA-NOT: v_mul_f16_sdwa

; SDWA-DAG: v_mul_f16_sdwa v[[DST_MUL_HI:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_f16_e32 v[[DST_MUL_LO:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; SDWA: v_or_b32_e32 v{{[0-9]+}}, v[[DST_MUL_HI]], v[[DST_MUL_LO]]
define amdgpu_kernel void @mul_v2half(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %ina, <2 x half> addrspace(1)* %inb) {
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

; SDWA-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v4half(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %ina, <4 x half> addrspace(1)* %inb) {
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

; SDWA-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_mul_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define amdgpu_kernel void @mul_v8half(<8 x half> addrspace(1)* %out, <8 x half> addrspace(1)* %ina, <8 x half> addrspace(1)* %inb) {
entry:
  %a = load <8 x half>, <8 x half> addrspace(1)* %ina, align 4
  %b = load <8 x half>, <8 x half> addrspace(1)* %inb, align 4
  %mul = fmul <8 x half> %a, %b
  store <8 x half> %mul, <8 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_i8:
; NOSDWA: v_mul_lo_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa
; SDWA: v_mul_lo_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SDWA-NOT: v_mul_u32_u24_sdwa

define amdgpu_kernel void @mul_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %ina, i8 addrspace(1)* %inb) {
entry:
  %a = load i8, i8 addrspace(1)* %ina, align 4
  %b = load i8, i8 addrspace(1)* %inb, align 4
  %mul = mul i8 %a, %b
  store i8 %mul, i8 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v2i8:
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA: v_mul_u32_u24_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1

define amdgpu_kernel void @mul_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(1)* %ina, <2 x i8> addrspace(1)* %inb) {
entry:
  %a = load <2 x i8>, <2 x i8> addrspace(1)* %ina, align 4
  %b = load <2 x i8>, <2 x i8> addrspace(1)* %inb, align 4
  %mul = mul <2 x i8> %a, %b
  store <2 x i8> %mul, <2 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v4i8:
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA-DAG: v_mul_u32_u24_sdwa
; SDWA-DAG: v_mul_u32_u24_sdwa
; SDWA-DAG: v_mul_u32_u24_sdwa

define amdgpu_kernel void @mul_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %ina, <4 x i8> addrspace(1)* %inb) {
entry:
  %a = load <4 x i8>, <4 x i8> addrspace(1)* %ina, align 4
  %b = load <4 x i8>, <4 x i8> addrspace(1)* %inb, align 4
  %mul = mul <4 x i8> %a, %b
  store <4 x i8> %mul, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_v8i8:
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA-DAG: v_mul_u32_u24_sdwa
; SDWA-DAG: v_mul_u32_u24_sdwa
; SDWA-DAG: v_mul_u32_u24_sdwa
; SDWA-DAG: v_mul_u32_u24_sdwa
; SDWA-DAG: v_mul_u32_u24_sdwa
; SDWA-DAG: v_mul_u32_u24_sdwa

define amdgpu_kernel void @mul_v8i8(<8 x i8> addrspace(1)* %out, <8 x i8> addrspace(1)* %ina, <8 x i8> addrspace(1)* %inb) {
entry:
  %a = load <8 x i8>, <8 x i8> addrspace(1)* %ina, align 4
  %b = load <8 x i8>, <8 x i8> addrspace(1)* %inb, align 4
  %mul = mul <8 x i8> %a, %b
  store <8 x i8> %mul, <8 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sitofp_v2i16_to_v2f16:
; NOSDWA-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; NOSDWA-DAG: v_ashrrev_i32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA-DAG: v_cvt_f32_i32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-DAG: v_cvt_f32_i32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_cvt_f32_i32_sdwa

; SDWA-DAG: v_cvt_f32_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0
; SDWA-DAG: v_cvt_f32_i32_sdwa v{{[0-9]+}}, sext(v{{[0-9]+}}) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

define amdgpu_kernel void @sitofp_v2i16_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %r.val = sitofp <2 x i16> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}


; GCN-LABEL: {{^}}mac_v2half:
; NOSDWA: v_lshrrev_b32_e32 v[[DST0:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v[[DST1:[0-9]+]], 16, v{{[0-9]+}}
; NOSDWA: v_mac_f16_e32 v[[DST_MAC:[0-9]+]], v[[DST1]], v[[DST0]]
; NOSDWA: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MAC]]
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v[[DST_SHL]], v{{[0-9]+}}
; NOSDWA-NOT: v_mac_f16_sdwa

; SDWA: v_mac_f16_sdwa v[[DST_MAC:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; SDWA: v_lshlrev_b32_e32 v[[DST_SHL:[0-9]+]], 16, v[[DST_MAC]]

define amdgpu_kernel void @mac_v2half(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %ina, <2 x half> addrspace(1)* %inb) {
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
; SDWA-NOT: v_mul_u32_u24_sdwa

define amdgpu_kernel void @immediate_mul_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
entry:
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %in, align 4
  %mul = mul <2 x i16> %a, <i16 123, i16 321>
  store <2 x i16> %mul, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; Double use of same src - should not convert it
; GCN-LABEL: {{^}}mulmul_v2i16:
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_mul_u32_u24_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; NOSDWA: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; NOSDWA-NOT: v_mul_u32_u24_sdwa

; SDWA: v_mul_u32_u24_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

define amdgpu_kernel void @mulmul_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %ina, <2 x i16> addrspace(1)* %inb) {
entry:
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %ina, align 4
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %inb, align 4
  %mul = mul <2 x i16> %a, %b
  %mul2 = mul <2 x i16> %mul, %b
  store <2 x i16> %mul2, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mul_add_shr_i32:
; NOSDWA-NOT: v_mul_u32_u24_sdwa
; NOSDWA-NOT: v_add_i32_sdwa
; SDWA-NOT: v_mul_u32_u24_sdwa
; SDWA-NOT: v_add_i32_sdwa

define void @mul_add_shr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %ina, i32 addrspace(1)* %inb, i1 addrspace(1)* %incond) {
entry:
  %a = load i32, i32 addrspace(1)* %ina, align 4
  %b = load i32, i32 addrspace(1)* %inb, align 4
  %cond = load i1, i1 addrspace(1)* %incond, align 4
  %shra = lshr i32 %a, 16
  %shrb = lshr i32 %b, 16
  br i1 %cond, label %mul_label, label %add_label
mul_label:
  %mul = mul i32 %shra, %shrb
  br label %store_label
add_label:
  %add = add i32 %shra, %shrb
  br label %store_label
store_label:
  %store = phi i32 [%mul, %mul_label], [%add, %add_label]
  store i32 %store, i32 addrspace(1)* %out, align 4
  ret void
}