; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}sample:
; GCN: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_cl:
; GCN: image_sample_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_d:
; GCN: image_sample_d {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_d(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.d.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_d_cl:
; GCN: image_sample_d_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_d_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.d.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_l:
; GCN: image_sample_l {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_l(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.l.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_b:
; GCN: image_sample_b {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_b(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.b.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_b_cl:
; GCN: image_sample_b_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_b_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.b.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_lz:
; GCN: image_sample_lz {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_lz(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.lz.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_cd:
; GCN: image_sample_cd {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_cd(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cd.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_cd_cl:
; GCN: image_sample_cd_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_cd_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c:
; GCN: image_sample_c {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_cl:
; GCN: image_sample_c_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_d:
; GCN: image_sample_c_d {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_d(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.d.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_d_cl:
; GCN: image_sample_c_d_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_d_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_l:
; GCN: image_sample_c_l {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_l(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.l.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_b:
; GCN: image_sample_c_b {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_b(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.b.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_b_cl:
; GCN: image_sample_c_b_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_b_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_lz:
; GCN: image_sample_c_lz {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_lz(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.lz.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_cd:
; GCN: image_sample_c_cd {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_cd(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cd.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_c_cd_cl:
; GCN: image_sample_c_cd_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xf
define void @sample_c_cd_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_f32:
; GCN: image_sample {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1
define void @sample_f32(float addrspace(1)* %out) {
main_body:
  %r = call float @llvm.amdgcn.image.sample.f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 0)
  store float %r, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sample_v2f32:
; GCN: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x3
define void @sample_v2f32(<2 x float> addrspace(1)* %out) {
main_body:
  %r = call <2 x float> @llvm.amdgcn.image.sample.v2f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 3, i1 0, i1 0, i1 0, i1 0, i1 0)
  store <2 x float> %r, <2 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_0:
; GCN: image_sample v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x1{{$}}
define void @adjust_writemask_sample_0(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt0 = extractelement <4 x float> %r, i32 0
  store float %elt0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_01:
; GCN: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x3{{$}}
define void @adjust_writemask_sample_01(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt0 = extractelement <4 x float> %r, i32 0
  %elt1 = extractelement <4 x float> %r, i32 1
  store volatile float %elt0, float addrspace(1)* %out
  store volatile float %elt1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_012:
; GCN: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x7{{$}}
define void @adjust_writemask_sample_012(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt0 = extractelement <4 x float> %r, i32 0
  %elt1 = extractelement <4 x float> %r, i32 1
  %elt2 = extractelement <4 x float> %r, i32 2
  store volatile float %elt0, float addrspace(1)* %out
  store volatile float %elt1, float addrspace(1)* %out
  store volatile float %elt2, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_12:
; GCN: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x6{{$}}
define void @adjust_writemask_sample_12(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt1 = extractelement <4 x float> %r, i32 1
  %elt2 = extractelement <4 x float> %r, i32 2
  store volatile float %elt1, float addrspace(1)* %out
  store volatile float %elt2, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_03:
; GCN: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x9{{$}}
define void @adjust_writemask_sample_03(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt0 = extractelement <4 x float> %r, i32 0
  %elt3 = extractelement <4 x float> %r, i32 3
  store volatile float %elt0, float addrspace(1)* %out
  store volatile float %elt3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_13:
; GCN: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xa{{$}}
define void @adjust_writemask_sample_13(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt1 = extractelement <4 x float> %r, i32 1
  %elt3 = extractelement <4 x float> %r, i32 3
  store volatile float %elt1, float addrspace(1)* %out
  store volatile float %elt3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_sample_123:
; GCN: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xe{{$}}
define void @adjust_writemask_sample_123(float addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt1 = extractelement <4 x float> %r, i32 1
  %elt2 = extractelement <4 x float> %r, i32 2
  %elt3 = extractelement <4 x float> %r, i32 3
  store volatile float %elt1, float addrspace(1)* %out
  store volatile float %elt2, float addrspace(1)* %out
  store volatile float %elt3, float addrspace(1)* %out
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.d.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.l.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.lz.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.cd.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0

declare <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.d.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.l.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0

declare float @llvm.amdgcn.image.sample.f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <2 x float> @llvm.amdgcn.image.sample.v2f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0

attributes #0 = { nounwind readnone }
