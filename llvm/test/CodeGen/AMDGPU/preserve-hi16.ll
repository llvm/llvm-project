; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}shl_i16:
; GCN: v_lshlrev_b16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @shl_i16(i16 %x, i16 %y) {
  %res = shl i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}lshr_i16:
; GCN: v_lshrrev_b16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @lshr_i16(i16 %x, i16 %y) {
  %res = lshr i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}ashr_i16:
; GCN: v_ashrrev_i16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @ashr_i16(i16 %x, i16 %y) {
  %res = ashr i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}add_u16:
; GCN: v_add_{{(nc_)*}}u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @add_u16(i16 %x, i16 %y) {
  %res = add i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}sub_u16:
; GCN: v_sub_{{(nc_)*}}u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @sub_u16(i16 %x, i16 %y) {
  %res = sub i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}mul_lo_u16:
; GCN: v_mul_lo_u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @mul_lo_u16(i16 %x, i16 %y) {
  %res = mul i16 %x, %y
  ret i16 %res
}

; GCN-LABEL: {{^}}min_u16:
; GCN: v_min_u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @min_u16(i16 %x, i16 %y) {
  %cmp = icmp ule i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}min_i16:
; GCN: v_min_i16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @min_i16(i16 %x, i16 %y) {
  %cmp = icmp sle i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}max_u16:
; GCN: v_max_u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @max_u16(i16 %x, i16 %y) {
  %cmp = icmp uge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}max_i16:
; GCN: v_max_i16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_setpc_b64
define i16 @max_i16(i16 %x, i16 %y) {
  %cmp = icmp sge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  ret i16 %res
}

; GCN-LABEL: {{^}}shl_i16_zext_i32:
; GCN: v_lshlrev_b16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @shl_i16_zext_i32(i16 %x, i16 %y) {
  %res = shl i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}lshr_i16_zext_i32:
; GCN: v_lshrrev_b16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @lshr_i16_zext_i32(i16 %x, i16 %y) {
  %res = lshr i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}ashr_i16_zext_i32:
; GCN: v_ashrrev_i16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @ashr_i16_zext_i32(i16 %x, i16 %y) {
  %res = ashr i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}add_u16_zext_i32:
; GCN: v_add_{{(nc_)*}}u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @add_u16_zext_i32(i16 %x, i16 %y) {
  %res = add i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}sub_u16_zext_i32:
; GCN: v_sub_{{(nc_)*}}u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @sub_u16_zext_i32(i16 %x, i16 %y) {
  %res = sub i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}mul_lo_u16_zext_i32:
; GCN: v_mul_lo_u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @mul_lo_u16_zext_i32(i16 %x, i16 %y) {
  %res = mul i16 %x, %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}min_u16_zext_i32:
; GCN: v_min_u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @min_u16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp ule i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}min_i16_zext_i32:
; GCN: v_min_i16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @min_i16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp sle i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}max_u16_zext_i32:
; GCN: v_max_u16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @max_u16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp uge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}

; GCN-LABEL: {{^}}max_i16_zext_i32:
; GCN: v_max_i16_e{{32|64}} [[OP:v[0-9]+]],
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[OP]]
; GCN-NEXT: s_setpc_b64
define i32 @max_i16_zext_i32(i16 %x, i16 %y) {
  %cmp = icmp sge i16 %x, %y
  %res = select i1 %cmp, i16 %x, i16 %y
  %zext = zext i16 %res to i32
  ret i32 %zext
}
