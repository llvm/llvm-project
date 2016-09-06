; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}gather4_v2:
; GCN: image_gather4 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_v2(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4:
; GCN: image_gather4 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_cl:
; GCN: image_gather4_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_l:
; GCN: image_gather4_l {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_l(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.l.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_b:
; GCN: image_gather4_b {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_b(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.b.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_b_cl:
; GCN: image_gather4_b_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_b_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_b_cl_v8:
; GCN: image_gather4_b_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_b_cl_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_lz_v2:
; GCN: image_gather4_lz {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_lz_v2(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.lz.v4f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_lz:
; GCN: image_gather4_lz {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_lz(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.lz.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}



; GCN-LABEL: {{^}}gather4_o:
; GCN: image_gather4_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_cl_o:
; GCN: image_gather4_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_cl_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.cl.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_cl_o_v8:
; GCN: image_gather4_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_cl_o_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.cl.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_l_o:
; GCN: image_gather4_l_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_l_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.l.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_l_o_v8:
; GCN: image_gather4_l_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_l_o_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.l.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_b_o:
; GCN: image_gather4_b_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_b_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.b.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_b_o_v8:
; GCN: image_gather4_b_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_b_o_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.b.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_b_cl_o:
; GCN: image_gather4_b_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_b_cl_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_lz_o:
; GCN: image_gather4_lz_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_lz_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.lz.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}gather4_c:
; GCN: image_gather4_c {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_cl:
; GCN: image_gather4_c_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_cl_v8:
; GCN: image_gather4_c_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_cl_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_l:
; GCN: image_gather4_c_l {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_l(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.l.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_l_v8:
; GCN: image_gather4_c_l {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_l_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.l.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_b:
; GCN: image_gather4_c_b {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_b(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.b.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_b_v8:
; GCN: image_gather4_c_b {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_b_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.b.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_b_cl:
; GCN: image_gather4_c_b_cl {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_b_cl(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_lz:
; GCN: image_gather4_c_lz {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_lz(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}gather4_c_o:
; GCN: image_gather4_c_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_o_v8:
; GCN: image_gather4_c_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_o_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_cl_o:
; GCN: image_gather4_c_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_cl_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_l_o:
; GCN: image_gather4_c_l_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_l_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.l.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_b_o:
; GCN: image_gather4_c_b_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_b_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.b.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_b_cl_o:
; GCN: image_gather4_c_b_cl_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_b_cl_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_lz_o:
; GCN: image_gather4_c_lz_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_lz_o(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.v4f32.v4f32.v8i32(<4 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}gather4_c_lz_o_v8:
; GCN: image_gather4_c_lz_o {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1 da
define void @gather4_c_lz_o_v8(<4 x float> addrspace(1)* %out) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.v4f32.v8f32.v8i32(<8 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 1, i1 0, i1 0, i1 0, i1 0, i1 1)
  store <4 x float> %r, <4 x float> addrspace(1)* %out
  ret void
}


declare <4 x float> @llvm.amdgcn.image.gather4.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.l.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.lz.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.lz.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0

declare <4 x float> @llvm.amdgcn.image.gather4.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.cl.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.cl.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.l.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.l.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.b.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.b.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0

declare <4 x float> @llvm.amdgcn.image.gather4.c.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.cl.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.cl.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.l.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.l.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.lz.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0

declare <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.l.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.v4f32.v8f32.v8i32(<8 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #0


attributes #0 = { nounwind readnone }
