; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}sample:
; GCN: image_sample_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_cl:
; GCN: image_sample_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_d:
; GCN: image_sample_d_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_d(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.d.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_d_cl:
; GCN: image_sample_d_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_d_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.d.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_l:
; GCN: image_sample_l_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_l(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.l.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_b:
; GCN: image_sample_b_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_b(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.b.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_b_cl:
; GCN: image_sample_b_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_b_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.b.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_lz:
; GCN: image_sample_lz_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_lz(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.lz.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_cd:
; GCN: image_sample_cd_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_cd(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cd.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_cd_cl:
; GCN: image_sample_cd_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_cd_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c:
; GCN: image_sample_c_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_cl:
; GCN: image_sample_c_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_d:
; GCN: image_sample_c_d_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_d(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.d.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_d_cl:
; GCN: image_sample_c_d_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_d_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_l:
; GCN: image_sample_c_l_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_l(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_b:
; GCN: image_sample_c_b_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_b(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.b.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_b_cl:
; GCN: image_sample_c_b_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_b_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_lz:
; GCN: image_sample_c_lz_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_lz(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.lz.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_cd:
; GCN: image_sample_c_cd_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_cd(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cd.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_cd_cl:
; GCN: image_sample_c_cd_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define amdgpu_kernel void @sample_c_cd_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_d_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_d_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.d.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_d_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_d_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.d.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_l_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_l_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.l.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_b_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_b_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.b.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_b_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_b_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.b.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_lz_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_lz_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.lz.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_cd_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_cd_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cd.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_cd_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_cd_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_d_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_d_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.d.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_d_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_d_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_l_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_l_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_b_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_b_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.b.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_b_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_b_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_lz_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_lz_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.lz.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_cd_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_cd_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cd.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_c_cd_cl_o_none_enabled:
; GCN-NOT: image
; GCN-NOT: store
define amdgpu_kernel void @adjust_writemask_sample_c_cd_cl_o_none_enabled(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.d.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.l.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.b.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.cd.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0

declare <4 x float> @llvm.amdgcn.image.sample.c.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.d.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.l.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.b.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0


attributes #0 = { nounwind readnone }
