; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: {{^}}load_1d:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @load_1d(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2d:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @load_2d(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_3d:
; GCN: image_load v[0:3], v[0:3], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @load_3d(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %r) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_cube:
; GCN: image_load v[0:3], v[0:3], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @load_cube(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1darray:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @load_1darray(<8 x i32> inreg %rsrc, i32 %s, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32 15, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darray:
; GCN: image_load v[0:3], v[0:3], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @load_2darray(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa:
; GCN: image_load v[0:3], v[0:3], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @load_2dmsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darraymsaa:
; GCN: image_load v[0:3], v[0:3], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @load_2darraymsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_1d:
; GCN: image_load_mip v[0:3], v[0:1], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @load_mip_1d(<8 x i32> inreg %rsrc, i32 %s, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32 15, i32 %s, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_2d:
; GCN: image_load_mip v[0:3], v[0:3], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @load_mip_2d(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_3d:
; GCN: image_load_mip v[0:3], v[0:3], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @load_mip_3d(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %r, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %r, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_cube:
; GCN: image_load_mip v[0:3], v[0:3], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @load_mip_cube(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_1darray:
; GCN: image_load_mip v[0:3], v[0:3], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @load_mip_1darray(<8 x i32> inreg %rsrc, i32 %s, i32 %slice, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32 15, i32 %s, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_mip_2darray:
; GCN: image_load_mip v[0:3], v[0:3], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @load_mip_2darray(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}store_1d:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps void @store_1d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2d:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps void @store_2d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t) {
main_body:
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_3d:
; GCN: image_store v[0:3], v[4:7], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps void @store_3d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %r) {
main_body:
  call void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_cube:
; GCN: image_store v[0:3], v[4:7], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps void @store_cube(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice) {
main_body:
  call void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_1darray:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps void @store_1darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %slice) {
main_body:
  call void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2darray:
; GCN: image_store v[0:3], v[4:7], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps void @store_2darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice) {
main_body:
  call void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2dmsaa:
; GCN: image_store v[0:3], v[4:7], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps void @store_2dmsaa(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %fragid) {
main_body:
  call void @llvm.amdgcn.image.store.2dmsaa.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_2darraymsaa:
; GCN: image_store v[0:3], v[4:7], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps void @store_2darraymsaa(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  call void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_1d:
; GCN: image_store_mip v[0:3], v[4:5], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps void @store_mip_1d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_2d:
; GCN: image_store_mip v[0:3], v[4:7], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps void @store_mip_2d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_3d:
; GCN: image_store_mip v[0:3], v[4:7], s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps void @store_mip_3d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %r, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %r, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_cube:
; GCN: image_store_mip v[0:3], v[4:7], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps void @store_mip_cube(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_1darray:
; GCN: image_store_mip v[0:3], v[4:7], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps void @store_mip_1darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %slice, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_mip_2darray:
; GCN: image_store_mip v[0:3], v[4:7], s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps void @store_mip_2darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}getresinfo_1d:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @getresinfo_1d(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2d:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @getresinfo_2d(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2d.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_3d:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @getresinfo_3d(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.3d.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_cube:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @getresinfo_cube(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.cube.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_1darray:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @getresinfo_1darray(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.1darray.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2darray:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @getresinfo_2darray(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2darray.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2dmsaa:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm{{$}}
define amdgpu_ps <4 x float> @getresinfo_2dmsaa(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2dmsaa.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}getresinfo_2darraymsaa:
; GCN: image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm da{{$}}
define amdgpu_ps <4 x float> @getresinfo_2darraymsaa(<8 x i32> inreg %rsrc, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2darraymsaa.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_V1:
; GCN: image_load v0, v0, s[0:7] dmask:0x8 unorm{{$}}
define amdgpu_ps float @load_1d_V1(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call float @llvm.amdgcn.image.load.1d.f32.i32(i32 8, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret float %v
}

; GCN-LABEL: {{^}}load_1d_V2:
; GCN: image_load v[0:1], v0, s[0:7] dmask:0x9 unorm{{$}}
define amdgpu_ps <2 x float> @load_1d_V2(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <2 x float> @llvm.amdgcn.image.load.1d.v2f32.i32(i32 9, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <2 x float> %v
}

; GCN-LABEL: {{^}}store_1d_V1:
; GCN: image_store v0, v1, s[0:7] dmask:0x2 unorm{{$}}
define amdgpu_ps void @store_1d_V1(<8 x i32> inreg %rsrc, float %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.f32.i32(float %vdata, i32 2, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store_1d_V2:
; GCN: image_store v[0:1], v2, s[0:7] dmask:0xc unorm{{$}}
define amdgpu_ps void @store_1d_V2(<8 x i32> inreg %rsrc, <2 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v2f32.i32(<2 x float> %vdata, i32 12, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}load_1d_glc:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm glc{{$}}
define amdgpu_ps <4 x float> @load_1d_glc(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 1)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_slc:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm slc{{$}}
define amdgpu_ps <4 x float> @load_1d_slc(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 2)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_glc_slc:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm glc slc{{$}}
define amdgpu_ps <4 x float> @load_1d_glc_slc(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 3)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}store_1d_glc:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm glc{{$}}
define amdgpu_ps void @store_1d_glc(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 1)
  ret void
}

; GCN-LABEL: {{^}}store_1d_slc:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm slc{{$}}
define amdgpu_ps void @store_1d_slc(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 2)
  ret void
}

; GCN-LABEL: {{^}}store_1d_glc_slc:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm glc slc{{$}}
define amdgpu_ps void @store_1d_glc_slc(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %vdata, i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 3)
  ret void
}

; GCN-LABEL: {{^}}getresinfo_dmask0:
; GCN-NOT: image
; GCN: ; return to shader part epilog
define amdgpu_ps <4 x float> @getresinfo_dmask0(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %mip) #0 {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 0, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %r
}

; Ideally, the register allocator would avoid the wait here
;
; GCN-LABEL: {{^}}image_store_wait:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm
; SI: s_waitcnt expcnt(0)
; GCN: image_load v[0:3], v4, s[8:15] dmask:0xf unorm
; GCN: s_waitcnt vmcnt(0)
; GCN: image_store v[0:3], v4, s[16:23] dmask:0xf unorm
define amdgpu_ps void @image_store_wait(<8 x i32> inreg %arg, <8 x i32> inreg %arg1, <8 x i32> inreg %arg2, <4 x float> %arg3, i32 %arg4) #0 {
main_body:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %arg3, i32 15, i32 %arg4, <8 x i32> %arg, i32 0, i32 0)
  %data = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %arg4, <8 x i32> %arg1, i32 0, i32 0)
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %data, i32 15, i32 %arg4, <8 x i32> %arg2, i32 0, i32 0)
  ret void
}

; SI won't merge ds memory operations, because of the signed offset bug, so
; we only have check lines for VI.
; VI-LABEL: image_load_mmo
; VI: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; VI: ds_write2_b32 v{{[0-9]+}}, [[ZERO]], [[ZERO]] offset1:4
define amdgpu_ps float @image_load_mmo(<8 x i32> inreg %rsrc, float addrspace(3)* %lds, <2 x i32> %c) #0 {
  store float 0.000000e+00, float addrspace(3)* %lds
  %c0 = extractelement <2 x i32> %c, i32 0
  %c1 = extractelement <2 x i32> %c, i32 1
  %tex = call float @llvm.amdgcn.image.load.2d.f32.i32(i32 15, i32 %c0, i32 %c1, <8 x i32> %rsrc, i32 0, i32 0)
  %tmp2 = getelementptr float, float addrspace(3)* %lds, i32 4
  store float 0.000000e+00, float addrspace(3)* %tmp2
  ret float %tex
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2dmsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0

declare void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0

declare <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.3d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.cube.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.1darray.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2darray.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2dmsaa.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2
declare <4 x float> @llvm.amdgcn.image.getresinfo.2darraymsaa.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #2

declare float @llvm.amdgcn.image.load.1d.f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare float @llvm.amdgcn.image.load.2d.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.load.1d.v2f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare void @llvm.amdgcn.image.store.1d.f32.i32(float, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.1d.v2f32.i32(<2 x float>, i32, i32, <8 x i32>, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
