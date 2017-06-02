; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}image_load_v4i32:
; GCN: image_load v[0:3], v[0:3], s[0:7] dmask:0xf unorm
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps <4 x float> @image_load_v4i32(<8 x i32> inreg %rsrc, <4 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v4f32.v4i32.v8i32(<4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret <4 x float> %tex
}

; GCN-LABEL: {{^}}image_load_v2i32:
; GCN: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps <4 x float> @image_load_v2i32(<8 x i32> inreg %rsrc, <2 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v4f32.v2i32.v8i32(<2 x i32> %c, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret <4 x float> %tex
}

; GCN-LABEL: {{^}}image_load_i32:
; GCN: image_load v[0:3], v0, s[0:7] dmask:0xf unorm
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps <4 x float> @image_load_i32(<8 x i32> inreg %rsrc, i32 %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v4f32.i32.v8i32(i32 %c, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret <4 x float> %tex
}

; GCN-LABEL: {{^}}image_load_mip:
; GCN: image_load_mip v[0:3], v[0:3], s[0:7] dmask:0xf unorm
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps <4 x float> @image_load_mip(<8 x i32> inreg %rsrc, <4 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.mip.v4f32.v4i32.v8i32(<4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret <4 x float> %tex
}

; GCN-LABEL: {{^}}image_load_1:
; GCN: image_load v0, v[0:3], s[0:7] dmask:0x1 unorm
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_load_1(<8 x i32> inreg %rsrc, <4 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v4f32.v4i32.v8i32(<4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  %elt = extractelement <4 x float> %tex, i32 0
  ret float %elt
}

; GCN-LABEL: {{^}}image_load_f32_v2i32:
; GCN: image_load {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 unorm
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_load_f32_v2i32(<8 x i32> inreg %rsrc, <2 x i32> %c) #0 {
main_body:
  %tex = call float @llvm.amdgcn.image.load.f32.v2i32.v8i32(<2 x i32> %c, <8 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false)
  ret float %tex
}

; GCN-LABEL: {{^}}image_load_v2f32_v4i32:
; GCN: image_load {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x3 unorm
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps <2 x float> @image_load_v2f32_v4i32(<8 x i32> inreg %rsrc, <4 x i32> %c) #0 {
main_body:
  %tex = call <2 x float> @llvm.amdgcn.image.load.v2f32.v4i32.v8i32(<4 x i32> %c, <8 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false)
  ret <2 x float> %tex
}

; GCN-LABEL: {{^}}image_store_v4i32:
; GCN: image_store v[0:3], v[4:7], s[0:7] dmask:0xf unorm
define amdgpu_ps void @image_store_v4i32(<8 x i32> inreg %rsrc, <4 x float> %data, <4 x i32> %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.v4f32.v4i32.v8i32(<4 x float> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}image_store_v2i32:
; GCN: image_store v[0:3], v[4:5], s[0:7] dmask:0xf unorm
define amdgpu_ps void @image_store_v2i32(<8 x i32> inreg %rsrc, <4 x float> %data, <2 x i32> %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.v4f32.v2i32.v8i32(<4 x float> %data, <2 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}image_store_i32:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm
define amdgpu_ps void @image_store_i32(<8 x i32> inreg %rsrc, <4 x float> %data, i32 %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.v4f32.i32.v8i32(<4 x float> %data, i32 %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}image_store_f32_i32:
; GCN: image_store {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 unorm
define amdgpu_ps void @image_store_f32_i32(<8 x i32> inreg %rsrc, float %data, i32 %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.f32.i32.v8i32(float %data, i32 %coords, <8 x i32> %rsrc, i32 1, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}image_store_v2f32_v4i32:
; GCN: image_store {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x3 unorm
define amdgpu_ps void @image_store_v2f32_v4i32(<8 x i32> inreg %rsrc, <2 x float> %data, <4 x i32> %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.v2f32.v4i32.v8i32(<2 x float> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 3, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}image_store_mip:
; GCN: image_store_mip v[0:3], v[4:7], s[0:7] dmask:0xf unorm
define amdgpu_ps void @image_store_mip(<8 x i32> inreg %rsrc, <4 x float> %data, <4 x i32> %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.mip.v4f32.v4i32.v8i32(<4 x float> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; GCN-LABEL: {{^}}getresinfo:
; GCN: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_ps void @getresinfo() #0 {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getresinfo.v4f32.i32.v8i32(i32 undef, <8 x i32> undef, i32 15, i1 false, i1 false, i1 false, i1 false)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %r0, float %r1, float %r2, float %r3, i1 true, i1 true) #0
  ret void
}

; Ideally, the register allocator would avoid the wait here
;
; GCN-LABEL: {{^}}image_store_wait:
; GCN: image_store v[0:3], v4, s[0:7] dmask:0xf unorm
; GCN: s_waitcnt expcnt(0)
; GCN: image_load v[0:3], v4, s[8:15] dmask:0xf unorm
; GCN: s_waitcnt vmcnt(0)
; GCN: image_store v[0:3], v4, s[16:23] dmask:0xf unorm
define amdgpu_ps void @image_store_wait(<8 x i32> inreg %arg, <8 x i32> inreg %arg1, <8 x i32> inreg %arg2, <4 x float> %arg3, i32 %arg4) #0 {
main_body:
  call void @llvm.amdgcn.image.store.v4f32.i32.v8i32(<4 x float> %arg3, i32 %arg4, <8 x i32> %arg, i32 15, i1 false, i1 false, i1 false, i1 false)
  %data = call <4 x float> @llvm.amdgcn.image.load.v4f32.i32.v8i32(i32 %arg4, <8 x i32> %arg1, i32 15, i1 false, i1 false, i1 false, i1 false)
  call void @llvm.amdgcn.image.store.v4f32.i32.v8i32(<4 x float> %data, i32 %arg4, <8 x i32> %arg2, i32 15, i1 false, i1 false, i1 false, i1 false)
  ret void
}

; SI won't merge ds memory operations, because of the signed offset bug, so
; we only have check lines for VI.
; VI-LABEL: image_load_mmo
; VI: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; VI: ds_write2_b32 v{{[0-9]+}}, [[ZERO]], [[ZERO]] offset1:4
define amdgpu_ps void @image_load_mmo(float addrspace(3)* %lds, <2 x i32> %c, <8 x i32> inreg %rsrc) #0 {
bb:
  store float 0.000000e+00, float addrspace(3)* %lds
  %tex = call float @llvm.amdgcn.image.load.f32.v2i32.v8i32(<2 x i32> %c, <8 x i32> %rsrc, i32 15, i1 false, i1 false, i1 false, i1 false)
  %tmp2 = getelementptr float, float addrspace(3)* %lds, i32 4
  store float 0.000000e+00, float addrspace(3)* %tmp2
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tex, float %tex, float %tex, float %tex, i1 true, i1 true) #0
  ret void
}

declare float @llvm.amdgcn.image.load.f32.v2i32.v8i32(<2 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare <2 x float> @llvm.amdgcn.image.load.v2f32.v4i32.v8i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare void @llvm.amdgcn.image.store.f32.i32.v8i32(float, i32, <8 x i32>, i32, i1, i1, i1, i1) #0


declare void @llvm.amdgcn.image.store.v2f32.v4i32.v8i32(<2 x float>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #0
declare void @llvm.amdgcn.image.store.v4f32.i32.v8i32(<4 x float>, i32, <8 x i32>, i32, i1, i1, i1, i1) #0
declare void @llvm.amdgcn.image.store.v4f32.v2i32.v8i32(<4 x float>, <2 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #0
declare void @llvm.amdgcn.image.store.v4f32.v4i32.v8i32(<4 x float>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #0
declare void @llvm.amdgcn.image.store.mip.v4f32.v4i32.v8i32(<4 x float>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #0

declare <4 x float> @llvm.amdgcn.image.load.v4f32.i32.v8i32(i32, <8 x i32>, i32, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.load.v4f32.v2i32.v8i32(<2 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.load.v4f32.v4i32.v8i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.v4f32.v4i32.v8i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.v4f32.i32.v8i32(i32, <8 x i32>, i32, i1, i1, i1, i1) #1

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
