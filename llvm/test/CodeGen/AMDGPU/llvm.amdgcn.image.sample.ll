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


attributes #0 = { nounwind readnone }
