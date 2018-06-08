; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=UNPACKED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX81 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}image_load_f16
; GCN: image_load v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x1 unorm d16
define half @image_load_f16(<4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  %tex = call half @llvm.amdgcn.image.load.f16.v4i32.v8i32(<4 x i32> %coords, <8 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false)
  ret half %tex
}

; GCN-LABEL: {{^}}image_load_v2f16:
; UNPACKED: image_load v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x3 unorm d16
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: image_load v[[HI:[0-9]+]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x3 unorm d16
; PACKED: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v[[HI]]
define half @image_load_v2f16(<4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  %tex = call <2 x half> @llvm.amdgcn.image.load.v2f16.v4i32.v8i32(<4 x i32> %coords, <8 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false)
  %elt = extractelement <2 x half> %tex, i32 1
  ret half %elt
}

; GCN-LABEL: {{^}}image_load_v4f16:
; UNPACKED: image_load v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: image_load v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16
; PACKED: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v[[HI]]
define half @image_load_v4f16(<4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.load.v4f16.v4i32.v8i32(<4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  %elt = extractelement <4 x half> %tex, i32 3
  ret half %elt
}

; GCN-LABEL: {{^}}image_load_mip_v4f16:
; UNPACKED: image_load_mip v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: image_load_mip v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16
; PACKED: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v[[HI]]
define half @image_load_mip_v4f16(<4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.load.mip.v4f16.v4i32.v8i32(<4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  %elt = extractelement <4 x half> %tex, i32 3
  ret half %elt
}

; GCN-LABEL: {{^}}image_store_f16
; GCN: s_load_dword s[[LO:[0-9]+]],
; GCN: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[LO]]
; GCN: image_store v[[V_LO]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x1 unorm d16
define amdgpu_kernel void @image_store_f16(half %data, <4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  call void @llvm.amdgcn.image.store.f16.v4i32.v8i32(half %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; FIXME: Eliminate and to get low bits
; GCN-LABEL: {{^}}image_store_v2f16:
; UNPACKED: s_load_dword [[DATA:s[0-9]+]]
; UNPACKED-DAG: s_lshr_b32 [[UNPACK_1:s[0-9]+]], [[DATA]], 16
; UNPACKED-DAG: s_and_b32 [[UNPACK_0:s[0-9]+]], [[DATA]], 0xffff
; UNPACKED-DAG: v_mov_b32_e32 v[[V_UNPACK_0:[0-9]+]], [[UNPACK_0]]
; UNPACKED-DAG: v_mov_b32_e32 v[[V_UNPACK_1:[0-9]+]], [[UNPACK_1]]



; UNPACKED: image_store v{{\[}}[[V_UNPACK_0]]:[[V_UNPACK_1]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x3 unorm d16

; PACKED: image_store v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x3 unorm d16
define amdgpu_kernel void @image_store_v2f16(<2 x half> %data, <4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  call void @llvm.amdgcn.image.store.v2f16.v4i32.v8i32(<2 x half> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}image_store_v4f16
; UNPACKED: s_load_dword s
; UNPACKED: s_load_dword s
; UNPACKED: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; UNPACKED: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; UNPACKED: s_and_b32
; UNPACKED: s_and_b32
; UNPACKED: image_store v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16

; PACKED: s_load_dword [[DATA0:s[0-9]+]]
; PACKED: s_load_dword [[DATA1:s[0-9]+]]
; PACKED: v_mov_b32_e32 v[[V_LO:[0-9]+]], [[DATA0]]
; PACKED: v_mov_b32_e32 v[[V_HI:[0-9]+]], [[DATA1]]
; PACKED: image_store v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16
define amdgpu_kernel void @image_store_v4f16(<4 x half> %data, <4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  call void @llvm.amdgcn.image.store.v4f16.v4i32.v8i32(<4 x half> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}image_store_mip_v4f16
; UNPACKD: s_load_dword s
; UNPACKD: s_load_dword s
; UNPACKED: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; UNPACKED: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; UNPACKED: s_and_b32
; UNPACKED: s_and_b32
; UNPACKED: image_store_mip v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16

; PACKED: s_load_dword [[DATA0:s[0-9]+]]
; PACKED: s_load_dword [[DATA1:s[0-9]+]]
; PACKED: v_mov_b32_e32 v[[V_LO:[0-9]+]], [[DATA0]]
; PACKED: v_mov_b32_e32 v[[V_HI:[0-9]+]], [[DATA1]]
; PACKED: image_store_mip v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf unorm d16
define amdgpu_kernel void @image_store_mip_v4f16(<4 x half> %data, <4 x i32> %coords, <8 x i32> inreg %rsrc) {
main_body:
  call void @llvm.amdgcn.image.store.mip.v4f16.v4i32.v8i32(<4 x half> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret void
}


declare half @llvm.amdgcn.image.load.f16.v4i32.v8i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)
declare <2 x half> @llvm.amdgcn.image.load.v2f16.v4i32.v8i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)
declare <4 x half> @llvm.amdgcn.image.load.v4f16.v4i32.v8i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)
declare <4 x half> @llvm.amdgcn.image.load.mip.v4f16.v4i32.v8i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)

declare void @llvm.amdgcn.image.store.f16.v4i32.v8i32(half, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)
declare void @llvm.amdgcn.image.store.v2f16.v4i32.v8i32(<2 x half>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)
declare void @llvm.amdgcn.image.store.v4f16.v4i32.v8i32(<4 x half>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)
declare void @llvm.amdgcn.image.store.mip.v4f16.v4i32.v8i32(<4 x half>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1)
