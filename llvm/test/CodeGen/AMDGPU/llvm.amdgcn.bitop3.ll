; RUN: llc -march=amdgcn -mcpu=gfx1210 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-SDAG %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1210 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-GISEL %s

declare i32 @llvm.amdgcn.bitop3.i32(i32, i32, i32, i8)
declare i16 @llvm.amdgcn.bitop3.i16(i16, i16, i16, i8)

; GCN-LABEL: {{^}}bitop3_b32_vvv:
; GCN: v_bitop3_b32 v0, v0, v1, v2 bitop3:0xf
define amdgpu_ps float @bitop3_b32_vvv(i32 %a, i32 %b, i32 %c) {
  %ret = call i32 @llvm.amdgcn.bitop3.i32(i32 %a, i32 %b, i32 %c, i8 15)
  %ret_cast = bitcast i32 %ret to float
  ret float %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b32_svv:
; GCN: v_bitop3_b32 v0, s0, v0, v1 bitop3:0x10
define amdgpu_ps float @bitop3_b32_svv(i32 inreg %a, i32 %b, i32 %c) {
  %ret = call i32 @llvm.amdgcn.bitop3.i32(i32 %a, i32 %b, i32 %c, i8 16)
  %ret_cast = bitcast i32 %ret to float
  ret float %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b32_ssv:
; GCN: v_bitop3_b32 v0, s0, s1, v0 bitop3:0x11
define amdgpu_ps float @bitop3_b32_ssv(i32 inreg %a, i32 inreg %b, i32 %c) {
  %ret = call i32 @llvm.amdgcn.bitop3.i32(i32 %a, i32 %b, i32 %c, i8 17)
  %ret_cast = bitcast i32 %ret to float
  ret float %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b32_sss:
; GCN: v_bitop3_b32 v0, s0, s1, v{{[0-9]+}} bitop3:0x12
define amdgpu_ps float @bitop3_b32_sss(i32 inreg %a, i32 inreg %b, i32 inreg %c) {
  %ret = call i32 @llvm.amdgcn.bitop3.i32(i32 %a, i32 %b, i32 %c, i8 18)
  %ret_cast = bitcast i32 %ret to float
  ret float %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b32_vvi:
; GCN: v_bitop3_b32 v0, v0, v1, 0x3e8 bitop3:0x13
define amdgpu_ps float @bitop3_b32_vvi(i32 %a, i32 %b) {
  %ret = call i32 @llvm.amdgcn.bitop3.i32(i32 %a, i32 %b, i32 1000, i8 19)
  %ret_cast = bitcast i32 %ret to float
  ret float %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b32_vii:
; GCN-SDAG:  v_bitop3_b32 v0, v0, s{{[0-9]+}}, 0x3e8 bitop3:0x14
; GCN-GISEL: v_bitop3_b32 v0, v0, 0x7d0, {{[vs][0-9]+}} bitop3:0x14
define amdgpu_ps float @bitop3_b32_vii(i32 %a) {
  %ret = call i32 @llvm.amdgcn.bitop3.i32(i32 %a, i32 2000, i32 1000, i8 20)
  %ret_cast = bitcast i32 %ret to float
  ret float %ret_cast
}

; FIXME: Constant fold this

; GCN-LABEL: {{^}}bitop3_b32_iii:
; GCN: v_bitop3_b32
define amdgpu_ps float @bitop3_b32_iii() {
  %ret = call i32 @llvm.amdgcn.bitop3.i32(i32 3000, i32 2000, i32 1000, i8 21)
  %ret_cast = bitcast i32 %ret to float
  ret float %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b16_vvv:
; GCN: v_bitop3_b16 v0, v0, v1, v2 bitop3:0xf
define amdgpu_ps half @bitop3_b16_vvv(i16 %a, i16 %b, i16 %c) {
  %ret = call i16 @llvm.amdgcn.bitop3.i16(i16 %a, i16 %b, i16 %c, i8 15)
  %ret_cast = bitcast i16 %ret to half
  ret half %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b16_svv:
; GCN: v_bitop3_b16 v0, s0, v0, v1 bitop3:0x10
define amdgpu_ps half @bitop3_b16_svv(i16 inreg %a, i16 %b, i16 %c) {
  %ret = call i16 @llvm.amdgcn.bitop3.i16(i16 %a, i16 %b, i16 %c, i8 16)
  %ret_cast = bitcast i16 %ret to half
  ret half %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b16_ssv:
; GCN: v_bitop3_b16 v0, s0, s1, v0 bitop3:0x11
define amdgpu_ps half @bitop3_b16_ssv(i16 inreg %a, i16 inreg %b, i16 %c) {
  %ret = call i16 @llvm.amdgcn.bitop3.i16(i16 %a, i16 %b, i16 %c, i8 17)
  %ret_cast = bitcast i16 %ret to half
  ret half %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b16_sss:
; GCN: v_bitop3_b16 v0, s0, s1, v{{[0-9]+}} bitop3:0x12
define amdgpu_ps half @bitop3_b16_sss(i16 inreg %a, i16 inreg %b, i16 inreg %c) {
  %ret = call i16 @llvm.amdgcn.bitop3.i16(i16 %a, i16 %b, i16 %c, i8 18)
  %ret_cast = bitcast i16 %ret to half
  ret half %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b16_vvi:
; GCN: v_bitop3_b16 v0, v0, v1, 0x3e8 bitop3:0x13
define amdgpu_ps half @bitop3_b16_vvi(i16 %a, i16 %b) {
  %ret = call i16 @llvm.amdgcn.bitop3.i16(i16 %a, i16 %b, i16 1000, i8 19)
  %ret_cast = bitcast i16 %ret to half
  ret half %ret_cast
}

; GCN-LABEL: {{^}}bitop3_b16_vii:
; GCN-SDAG:  v_bitop3_b16 v0, v0, s{{[0-9]+}}, 0x3e8 bitop3:0x14
; GCN-GISEL: v_bitop3_b16 v0, v0, 0x7d0, {{[vs][0-9]+}} bitop3:0x14
define amdgpu_ps half @bitop3_b16_vii(i16 %a) {
  %ret = call i16 @llvm.amdgcn.bitop3.i16(i16 %a, i16 2000, i16 1000, i8 20)
  %ret_cast = bitcast i16 %ret to half
  ret half %ret_cast
}

; FIXME: Constant fold this

; GCN-LABEL: {{^}}bitop3_b16_iii:
; GCN: v_bitop3_b16
define amdgpu_ps half @bitop3_b16_iii() {
  %ret = call i16 @llvm.amdgcn.bitop3.i16(i16 3000, i16 2000, i16 1000, i8 21)
  %ret_cast = bitcast i16 %ret to half
  ret half %ret_cast
}
