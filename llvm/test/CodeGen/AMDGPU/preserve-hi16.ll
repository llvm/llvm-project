; RUN: llc -march=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9,GFX900 %s
; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9,GFX906 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10Plus %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -amdgpu-enable-delay-alu=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10Plus %s

; GCN-LABEL: {{^}}shl_i16:
; GCN: v_lshlrev_b16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @shl_i16(i16 %x, i16 %y) {
  %res = shl i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}lshr_i16:
; GCN: v_lshrrev_b16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @lshr_i16(i16 %x, i16 %y) {
  %res = lshr i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}ashr_i16:
; GCN: v_ashrrev_i16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @ashr_i16(i16 %x, i16 %y) {
  %res = ashr i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}add_u16:
; GCN: v_add_{{(nc_)*}}u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @add_u16(i16 %x, i16 %y) {
  %res = add i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}sub_u16:
; GCN: v_sub_{{(nc_)*}}u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @sub_u16(i16 %x, i16 %y) {
  %res = sub i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}mul_lo_u16:
; GCN: v_mul_lo_u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @mul_lo_u16(i16 %x, i16 %y) {
  %res = mul i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}min_u16:
; GCN: v_min_u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @min_u16(i16 %x, i16 %y) {
  %cmp = icmp ule i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}min_i16:
; GCN: v_min_i16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @min_i16(i16 %x, i16 %y) {
  %cmp = icmp sle i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}max_u16:
; GCN: v_max_u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @max_u16(i16 %x, i16 %y) {
  %cmp = icmp uge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}max_i16:
; GCN: v_max_i16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GCN-NEXT: s_setpc_b64
define i16 @max_i16(i16 %x, i16 %y) {
  %cmp = icmp sge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}shl_i16_zext_i32:
; GCN: v_lshlrev_b16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @shl_i16_zext_i32(i16 %x, i16 %y) {
  %res = shl i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}lshr_i16_zext_i32:
; GCN: v_lshrrev_b16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @lshr_i16_zext_i32(i16 %x, i16 %y) {
  %res = lshr i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}ashr_i16_zext_i32:
; GCN: v_ashrrev_i16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @ashr_i16_zext_i32(i16 %x, i16 %y) {
  %res = ashr i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}add_u16_zext_i32:
; GCN: v_add_{{(nc_)*}}u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @add_u16_zext_i32(i16 %x, i16 %y) {
  %res = add i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}sub_u16_zext_i32:
; GCN: v_sub_{{(nc_)*}}u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @sub_u16_zext_i32(i16 %x, i16 %y) {
  %res = sub i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}mul_lo_u16_zext_i32:
; GCN: v_mul_lo_u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @mul_lo_u16_zext_i32(i16 %x, i16 %y) {
  %res = mul i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}min_u16_zext_i32:
; GCN: v_min_u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @min_u16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp ule i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}min_i16_zext_i32:
; GCN: v_min_i16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @min_i16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp sle i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}max_u16_zext_i32:
; GCN: v_max_u16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @max_u16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp uge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}max_i16_zext_i32:
; GCN: v_max_i16{{(_e32|_e64)?}} [[OP:v[0-9]+]],
; GFX10Plus-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @max_i16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp sge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}zext_fadd_f16:
; GFX8: v_add_f16_e32 [[ADD:v[0-9]+]], v0, v1
; GFX8-NEXT: s_setpc_b64

; GFX9: v_add_f16_e32 [[ADD:v[0-9]+]], v0, v1
; GFX9-NEXT: s_setpc_b64

; GFX10Plus: v_add_f16_e32 [[ADD:v[0-9]+]], v0, v1
; GFX10Plus-NEXT: v_and_b32_e32 v0, 0xffff, [[ADD]]
define i32 @zext_fadd_f16(half %x, half %y) {
  %add = fadd half %x, %y
  %cast = bitcast half %add to i16
  %zext = zext i16 %cast to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}zext_fma_f16:
; GFX8: v_fma_f16 [[FMA:v[0-9]+]], v0, v1, v2
; GFX8-NEXT: s_setpc_b64

; GFX9: v_fma_f16 [[FMA:v[0-9]+]], v0, v1, v2
; GFX9-NEXT: v_and_b32_e32 v0, 0xffff, [[FMA]]

; GFX10Plus: v_fmac_f16_e32 [[FMA:v[0-9]+]], v0, v1
; GFX10Plus-NEXT: v_and_b32_e32 v0, 0xffff, [[FMA]]
define i32 @zext_fma_f16(half %x, half %y, half %z) {
  %fma = call half @llvm.fma.f16(half %x, half %y, half %z)
  %cast = bitcast half %fma to i16
  %zext = zext i16 %cast to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}zext_div_fixup_f16:
; GFX8: v_div_fixup_f16 v0, v0, v1, v2
; GFX8-NEXT: s_setpc_b64

; GFX9: v_div_fixup_f16 v0, v0, v1, v2
; GFX9-NEXT: v_and_b32_e32 v0, 0xffff, v0

; GFX10Plus: v_div_fixup_f16 v0, v0, v1, v2
; GFX10Plus-NEXT: v_and_b32_e32 v0, 0xffff, v0
define i32 @zext_div_fixup_f16(half %x, half %y, half %z) {
  %div.fixup = call half @llvm.amdgcn.div.fixup.f16(half %x, half %y, half %z)
  %cast = bitcast half %div.fixup to i16
  %zext = zext i16 %cast to i32
  ret i32 %zext
}

; We technically could eliminate the and on gfx9 here but we don't try
; to inspect the source of the fptrunc. We're only worried about cases
; that lower to v_fma_mix* instructions.

; GCN-LABEL: {{^}}zext_fptrunc_f16:
; GFX8: v_cvt_f16_f32_e32 v0, v0
; GFX8-NEXT: s_setpc_b64

; GFX9: v_cvt_f16_f32_e32 v0, v0
; GFX9-NEXT: s_setpc_b64

; GFX10Plus: v_cvt_f16_f32_e32 v0, v0
; GFX10Plus-NEXT: v_and_b32_e32 v0, 0xffff, v0
define i32 @zext_fptrunc_f16(float %x) {
  %fptrunc = fptrunc float %x to half
  %cast = bitcast half %fptrunc to i16
  %zext = zext i16 %cast to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}zext_fptrunc_fma_f16:
; GFX8: v_fma_f32 v0, v0, v1, v2
; GFX8-NEXT: v_cvt_f16_f32_e32 v0, v0
; GFX8-NEXT: s_setpc_b64

; GFX900: v_fma_f32 v0, v0, v1, v2
; GFX900-NEXT: v_cvt_f16_f32_e32 v0, v0
; GFX900-NEXT: s_setpc_b64

; GFX906: v_fma_mixlo_f16 v0, v0, v1, v2
; GFX906-NEXT: v_and_b32_e32 v0, 0xffff, v0

; GFX10Plus: v_fma_mixlo_f16 v0, v0, v1, v2
; GFX10Plus-NEXT: v_and_b32_e32 v0, 0xffff, v0
define i32 @zext_fptrunc_fma_f16(float %x, float %y, float %z) {
  %fma = call float @llvm.fma.f32(float %x, float %y, float %z)
  %fptrunc = fptrunc float %fma to half
  %cast = bitcast half %fptrunc to i16
  %zext = zext i16 %cast to i32
  ret i32 %zext
}

declare half @llvm.amdgcn.div.fixup.f16(half, half, half)
declare half @llvm.fma.f16(half, half, half)
declare float @llvm.fma.f32(float, float, float)
