; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}getlod:
; GCN: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf da
; GCN: s_waitcnt vmcnt(0)
; GCN: store_dwordx4
define amdgpu_kernel void @getlod(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.f32.v8i32(float undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}getlod_v2:
; GCN: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf da
; GCN: s_waitcnt vmcnt(0)
; GCN: store_dwordx4
define amdgpu_kernel void @getlod_v2(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}getlod_v4:
; GCN: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf da
; GCN: s_waitcnt vmcnt(0)
; GCN: store_dwordx4
define amdgpu_kernel void @getlod_v4(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_getlod_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_getlod_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

declare <4 x float> @llvm.amdgcn.image.getlod.v4f32.f32.v8i32(float, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.getlod.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.getlod.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0


attributes #0 = { nounwind readnone }
