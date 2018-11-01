; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: {{^}}load_1d:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @load_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2d:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @load_2d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %t = extractelement <2 x i16> %coords, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i16(i32 15, i16 %s, i16 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_3d:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @load_3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %r = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %r, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_cube:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @load_cube(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1darray:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @load_1darray(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %slice = extractelement <2 x i16> %coords, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i16(i32 15, i16 %s, i16 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darray:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @load_2darray(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @load_2dmsaa(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %fragid = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darraymsaa:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @load_2darraymsaa(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %fragid = extractelement <2 x i16> %coords_hi, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %slice, i16 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_1d:
; GCN: image_load_mip v[0:3], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @load_mip_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %mip = extractelement <2 x i16> %coords, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i16(i32 15, i16 %s, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_2d:
; GCN: image_load_mip v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @load_mip_2d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %mip = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_3d:
; GCN: image_load_mip v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @load_mip_3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %r = extractelement <2 x i16> %coords_hi, i32 0
  %mip = extractelement <2 x i16> %coords_hi, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %r, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_cube:
; GCN: image_load_mip v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @load_mip_cube(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %mip = extractelement <2 x i16> %coords_hi, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %slice, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_1darray:
; GCN: image_load_mip v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @load_mip_1darray(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %slice = extractelement <2 x i16> %coords_lo, i32 1
  %mip = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i16(i32 15, i16 %s, i16 %slice, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_2darray:
; GCN: image_load_mip v[0:3], v[0:1], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @load_mip_2darray(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %mip = extractelement <2 x i16> %coords_hi, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i16(i32 15, i16 %s, i16 %t, i16 %slice, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}store_1d:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store_1d(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2d:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store_2d(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %t = extractelement <2 x i16> %coords, i32 1
  call void @llvm.amdgcn.image.store.2d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_3d:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store_3d(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %r = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.3d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %r, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_cube:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps void @store_cube(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.cube.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_1darray:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps void @store_1darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %slice = extractelement <2 x i16> %coords, i32 1
  call void @llvm.amdgcn.image.store.1darray.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2darray:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps void @store_2darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.2darray.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2dmsaa:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store_2dmsaa(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %fragid = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.2dmsaa.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2darraymsaa:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps void @store_2darraymsaa(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %fragid = extractelement <2 x i16> %coords_hi, i32 1
  call void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %slice, i16 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_1d:
; GCN: image_store_mip v[0:3], v4, s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store_mip_1d(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %mip = extractelement <2 x i16> %coords, i32 1
  call void @llvm.amdgcn.image.store.mip.1d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_2d:
; GCN: image_store_mip v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store_mip_2d(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %mip = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.mip.2d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_3d:
; GCN: image_store_mip v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store_mip_3d(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %r = extractelement <2 x i16> %coords_hi, i32 0
  %mip = extractelement <2 x i16> %coords_hi, i32 1
  call void @llvm.amdgcn.image.store.mip.3d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %r, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_cube:
; GCN: image_store_mip v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps void @store_mip_cube(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %mip = extractelement <2 x i16> %coords_hi, i32 1
  call void @llvm.amdgcn.image.store.mip.cube.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %slice, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_1darray:
; GCN: image_store_mip v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps void @store_mip_1darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %slice = extractelement <2 x i16> %coords_lo, i32 1
  %mip = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.mip.1darray.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %slice, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_2darray:
; GCN: image_store_mip v[0:3], v[4:5], s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps void @store_mip_2darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %s = extractelement <2 x i16> %coords_lo, i32 0
  %t = extractelement <2 x i16> %coords_lo, i32 1
  %slice = extractelement <2 x i16> %coords_hi, i32 0
  %mip = extractelement <2 x i16> %coords_hi, i32 1
  call void @llvm.amdgcn.image.store.mip.2darray.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, i16 %t, i16 %slice, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}getresinfo_1d:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @getresinfo_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2d:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @getresinfo_2d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2d.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_3d:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @getresinfo_3d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.3d.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_cube:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @getresinfo_cube(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.cube.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_1darray:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @getresinfo_1darray(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.1darray.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2darray:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @getresinfo_2darray(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2darray.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2dmsaa:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps <4 x float> @getresinfo_2dmsaa(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2dmsaa.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2darraymsaa:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da{{$}}
define amdgpu_ps <4 x float> @getresinfo_2darraymsaa(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2darraymsaa.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_V1:
; GCN: image_load v0, v0, s[0:7] dmask:0x8 unorm a16
define amdgpu_ps float @load_1d_V1(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call float @llvm.amdgcn.image.load.1d.f32.i16(i32 8, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret float %v
}

; GCN-LABEL: {{^}}load_1d_V2:
; GCN: image_load v[0:1], v0, s[0:7] dmask:0x9 unorm a16
define amdgpu_ps <2 x float> @load_1d_V2(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call <2 x float> @llvm.amdgcn.image.load.1d.v2f32.i16(i32 9, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <2 x float> %v
}

; GCN-LABEL: {{^}}store_1d_V1:
; GCN: image_store v0, v1, s[0:7] dmask:0x2 unorm a16
define amdgpu_ps void @store_1d_V1(<8 x i32> inreg %rsrc, float %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.f32.i16(float %vdata, i32 2, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_1d_V2:
; GCN: image_store v[0:1], v2, s[0:7] dmask:0xc unorm a16
define amdgpu_ps void @store_1d_V2(<8 x i32> inreg %rsrc, <2 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v2f32.i16(<2 x float> %vdata, i32 12, i16 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}load_1d_glc:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm glc a16{{$}}
define amdgpu_ps <4 x float> @load_1d_glc(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 1)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_slc:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm slc a16{{$}}
define amdgpu_ps <4 x float> @load_1d_slc(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 2)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_glc_slc:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm glc slc a16{{$}}
define amdgpu_ps <4 x float> @load_1d_glc_slc(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 3)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}store_1d_glc:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm glc a16{{$}}
define amdgpu_ps void @store_1d_glc(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 1)
  ret void
}

; GCN-LABEL: {{^}}store_1d_slc:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm slc a16{{$}}
define amdgpu_ps void @store_1d_slc(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 2)
  ret void
}

; GCN-LABEL: {{^}}store_1d_glc_slc:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm glc slc a16{{$}}
define amdgpu_ps void @store_1d_glc_slc(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) {
main_body:
  %s = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %vdata, i32 15, i16 %s, <8 x i32> %rsrc, i32 0, i32 3)
  ret void
}

; GCN-LABEL: {{^}}getresinfo_dmask0:
; GCN-NOT: image
; GCN: ; return to shader part epilog
define amdgpu_ps <4 x float> @getresinfo_dmask0(<8 x i32> inreg %rsrc, <4 x float> %vdata, <2 x i16> %coords) #0 {
main_body:
  %mip = extractelement <2 x i16> %coords, i32 0
  %r = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i16(i32 0, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %r
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i16(i32, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i16(i32, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i16(i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i16(i32, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i16(i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i16(i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i16(i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #1

declare void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float>, i32, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2d.v4f32.i16(<4 x float>, i32, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.3d.v4f32.i16(<4 x float>, i32, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.cube.v4f32.i16(<4 x float>, i32, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.1darray.v4f32.i16(<4 x float>, i32, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2darray.v4f32.i16(<4 x float>, i32, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2dmsaa.v4f32.i16(<4 x float>, i32, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i16(<4 x float>, i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #0

declare void @llvm.amdgcn.image.store.mip.1d.v4f32.i16(<4 x float>, i32, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.2d.v4f32.i16(<4 x float>, i32, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.3d.v4f32.i16(<4 x float>, i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.cube.v4f32.i16(<4 x float>, i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.1darray.v4f32.i16(<4 x float>, i32, i16, i16, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.2darray.v4f32.i16(<4 x float>, i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #0

declare <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2d.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.3d.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.cube.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.1darray.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2darray.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2dmsaa.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2darraymsaa.v4f32.i16(i32, i16, <8 x i32>, i32, i32) #2

declare float @llvm.amdgcn.image.load.1d.f32.i16(i32, i16, <8 x i32>, i32, i32) #1
declare float @llvm.amdgcn.image.load.2d.f32.i16(i32, i16, i16, <8 x i32>, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.load.1d.v2f32.i16(i32, i16, <8 x i32>, i32, i32) #1
declare void @llvm.amdgcn.image.store.1d.f32.i16(float, i32, i16, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.1d.v2f32.i16(<2 x float>, i32, i16, <8 x i32>, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
