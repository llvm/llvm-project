; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX6789 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX6789 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}sample_1d:
; GFX6789: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_1d_tfe:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GCN: v_mov_b32_e32 v2, v0
; GCN: v_mov_b32_e32 v3, v0
; GCN: v_mov_b32_e32 v4, v0
; GFX6789: image_sample v[0:4], v5, s[0:7], s[8:11] dmask:0xf tfe{{$}}
; GFX10: image_sample v[0:4], v5, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <4 x float> @sample_1d_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 addrspace(1)* inreg %out, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}sample_1d_tfe_adjust_writemask_1:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GFX6789: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x1 tfe{{$}}
; GFX10: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <2 x float> @sample_1d_tfe_adjust_writemask_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 addrspace(1)* inreg %out, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %res.vec = extractvalue {<4 x float>,i32} %v, 0
  %res.f = extractelement <4 x float> %res.vec, i32 0
  %res.err = extractvalue {<4 x float>,i32} %v, 1
  %res.errf = bitcast i32 %res.err to float
  %res.tmp = insertelement <2 x float> undef, float %res.f, i32 0
  %res = insertelement <2 x float> %res.tmp, float %res.errf, i32 1
  ret <2 x float> %res
}

; GCN-LABEL: {{^}}sample_1d_tfe_adjust_writemask_2:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GFX6789: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x2 tfe{{$}}
; GFX10: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x2 dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <2 x float> @sample_1d_tfe_adjust_writemask_2(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %res.vec = extractvalue {<4 x float>,i32} %v, 0
  %res.f = extractelement <4 x float> %res.vec, i32 1
  %res.err = extractvalue {<4 x float>,i32} %v, 1
  %res.errf = bitcast i32 %res.err to float
  %res.tmp = insertelement <2 x float> undef, float %res.f, i32 0
  %res = insertelement <2 x float> %res.tmp, float %res.errf, i32 1
  ret <2 x float> %res
}

; GCN-LABEL: {{^}}sample_1d_tfe_adjust_writemask_3:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GFX6789: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x4 tfe{{$}}
; GFX10: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <2 x float> @sample_1d_tfe_adjust_writemask_3(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %res.vec = extractvalue {<4 x float>,i32} %v, 0
  %res.f = extractelement <4 x float> %res.vec, i32 2
  %res.err = extractvalue {<4 x float>,i32} %v, 1
  %res.errf = bitcast i32 %res.err to float
  %res.tmp = insertelement <2 x float> undef, float %res.f, i32 0
  %res = insertelement <2 x float> %res.tmp, float %res.errf, i32 1
  ret <2 x float> %res
}

; GCN-LABEL: {{^}}sample_1d_tfe_adjust_writemask_4:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GFX6789: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x8 tfe{{$}}
; GFX10: image_sample v[0:1], v2, s[0:7], s[8:11] dmask:0x8 dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <2 x float> @sample_1d_tfe_adjust_writemask_4(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %res.vec = extractvalue {<4 x float>,i32} %v, 0
  %res.f = extractelement <4 x float> %res.vec, i32 3
  %res.err = extractvalue {<4 x float>,i32} %v, 1
  %res.errf = bitcast i32 %res.err to float
  %res.tmp = insertelement <2 x float> undef, float %res.f, i32 0
  %res = insertelement <2 x float> %res.tmp, float %res.errf, i32 1
  ret <2 x float> %res
}

; GCN-LABEL: {{^}}sample_1d_tfe_adjust_writemask_12:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GCN: v_mov_b32_e32 v2, v0
; GFX6789: image_sample v[0:2], v3, s[0:7], s[8:11] dmask:0x3 tfe{{$}}
; GFX10: image_sample v[0:2], v3, s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <4 x float> @sample_1d_tfe_adjust_writemask_12(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %res.vec = extractvalue {<4 x float>,i32} %v, 0
  %res.f1 = extractelement <4 x float> %res.vec, i32 0
  %res.f2 = extractelement <4 x float> %res.vec, i32 1
  %res.err = extractvalue {<4 x float>,i32} %v, 1
  %res.errf = bitcast i32 %res.err to float
  %res.tmp1 = insertelement <4 x float> undef, float %res.f1, i32 0
  %res.tmp2 = insertelement <4 x float> %res.tmp1, float %res.f2, i32 1
  %res = insertelement <4 x float> %res.tmp2, float %res.errf, i32 2
  ret <4 x float> %res
}

; GCN-LABEL: {{^}}sample_1d_tfe_adjust_writemask_24:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GCN: v_mov_b32_e32 v2, v0
; GFX6789: image_sample v[0:2], v3, s[0:7], s[8:11] dmask:0xa tfe{{$}}
; GFX10: image_sample v[0:2], v3, s[0:7], s[8:11] dmask:0xa dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <4 x float> @sample_1d_tfe_adjust_writemask_24(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %res.vec = extractvalue {<4 x float>,i32} %v, 0
  %res.f1 = extractelement <4 x float> %res.vec, i32 1
  %res.f2 = extractelement <4 x float> %res.vec, i32 3
  %res.err = extractvalue {<4 x float>,i32} %v, 1
  %res.errf = bitcast i32 %res.err to float
  %res.tmp1 = insertelement <4 x float> undef, float %res.f1, i32 0
  %res.tmp2 = insertelement <4 x float> %res.tmp1, float %res.f2, i32 1
  %res = insertelement <4 x float> %res.tmp2, float %res.errf, i32 2
  ret <4 x float> %res
}

; GCN-LABEL: {{^}}sample_1d_tfe_adjust_writemask_134:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GCN: v_mov_b32_e32 v2, v0
; GCN: v_mov_b32_e32 v3, v0
; GFX6789: image_sample v[0:3], v4, s[0:7], s[8:11] dmask:0xd tfe{{$}}
; GFX10: image_sample v[0:3], v4, s[0:7], s[8:11] dmask:0xd dim:SQ_RSRC_IMG_1D tfe ;
define amdgpu_ps <4 x float> @sample_1d_tfe_adjust_writemask_134(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %res.vec = extractvalue {<4 x float>,i32} %v, 0
  %res.f1 = extractelement <4 x float> %res.vec, i32 0
  %res.f2 = extractelement <4 x float> %res.vec, i32 2
  %res.f3 = extractelement <4 x float> %res.vec, i32 3
  %res.err = extractvalue {<4 x float>,i32} %v, 1
  %res.errf = bitcast i32 %res.err to float
  %res.tmp1 = insertelement <4 x float> undef, float %res.f1, i32 0
  %res.tmp2 = insertelement <4 x float> %res.tmp1, float %res.f2, i32 1
  %res.tmp3 = insertelement <4 x float> %res.tmp2, float %res.f3, i32 2
  %res = insertelement <4 x float> %res.tmp3, float %res.errf, i32 3
  ret <4 x float> %res
}

; GCN-LABEL: {{^}}sample_1d_lwe:
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, v0
; GCN: v_mov_b32_e32 v2, v0
; GCN: v_mov_b32_e32 v3, v0
; GCN: v_mov_b32_e32 v4, v0
; GFX6789: image_sample v[0:4], v5, s[0:7], s[8:11] dmask:0xf lwe{{$}}
; GFX10: image_sample v[0:4], v5, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D lwe ;
define amdgpu_ps <4 x float> @sample_1d_lwe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 addrspace(1)* inreg %out, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 2, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}sample_2d:
; GFX6789: image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_3d:
; GFX6789: image_sample v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_3D ;
define amdgpu_ps <4 x float> @sample_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %r) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 15, float %s, float %t, float %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cube:
; GFX6789: image_sample v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf da{{$}}
; GFX10: image_sample v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_CUBE ;
define amdgpu_ps <4 x float> @sample_cube(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %face) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f32(i32 15, float %s, float %t, float %face, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_1darray:
; GFX6789: image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf da{{$}}
; GFX10: image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY ;
define amdgpu_ps <4 x float> @sample_1darray(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32 15, float %s, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_2darray:
; GFX6789: image_sample v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf da{{$}}
; GFX10: image_sample v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY ;
define amdgpu_ps <4 x float> @sample_2darray(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2darray.v4f32.f32(i32 15, float %s, float %t, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_1d:
; GFX6789: image_sample_c v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.1d.v4f32.f32(i32 15, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_2d:
; GFX6789: image_sample_c v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.2d.v4f32.f32(i32 15, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cl_1d:
; GFX6789: image_sample_cl v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_cl v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cl.1d.v4f32.f32(i32 15, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cl_2d:
; GFX6789: image_sample_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_cl v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cl.2d.v4f32.f32(i32 15, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cl_1d:
; GFX6789: image_sample_c_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_cl v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cl.1d.v4f32.f32(i32 15, float %zcompare, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cl_2d:
; GFX6789: image_sample_c_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cl.2d.v4f32.f32(i32 15, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_1d:
; GFX6789: image_sample_b v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_b v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_b_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.1d.v4f32.f32.f32(i32 15, float %bias, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_2d:
; GFX6789: image_sample_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_b_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.2d.v4f32.f32.f32(i32 15, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_1d:
; GFX6789: image_sample_c_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_b_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.1d.v4f32.f32.f32(i32 15, float %bias, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_2d:
; GFX6789: image_sample_c_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_b_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.2d.v4f32.f32.f32(i32 15, float %bias, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_cl_1d:
; GFX6789: image_sample_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_b_cl v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_b_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.cl.1d.v4f32.f32.f32(i32 15, float %bias, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_cl_2d:
; GFX6789: image_sample_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_b_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.cl.2d.v4f32.f32.f32(i32 15, float %bias, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_cl_1d:
; GFX6789: image_sample_c_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_b_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.1d.v4f32.f32.f32(i32 15, float %bias, float %zcompare, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_cl_2d:
; GFX6789: image_sample_c_b_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_b_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_b_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.2d.v4f32.f32.f32(i32 15, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_1d:
; GFX6789: image_sample_d v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_d v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_d_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32 15, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_2d:
; GFX6789: image_sample_d v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_d v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_d_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32 15, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_1d:
; GFX6789: image_sample_c_d v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_d v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_d_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_2d:
; GFX6789: image_sample_c_d v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_d v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_d_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.2d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_cl_1d:
; GFX6789: image_sample_d_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_d_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_d_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.cl.1d.v4f32.f32.f32(i32 15, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_cl_2d:
; GFX6789: image_sample_d_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_d_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_d_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.cl.2d.v4f32.f32.f32(i32 15, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_cl_1d:
; GFX6789: image_sample_c_d_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_d_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_d_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_cl_2d:
; GFX6789: image_sample_c_d_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_d_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_d_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.2d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_1d:
; GFX6789: image_sample_cd v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_cd v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_cd_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.1d.v4f32.f32.f32(i32 15, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_2d:
; GFX6789: image_sample_cd v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_cd v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_cd_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.2d.v4f32.f32.f32(i32 15, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_1d:
; GFX6789: image_sample_c_cd v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_cd v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_cd_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_2d:
; GFX6789: image_sample_c_cd v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_cd v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_cd_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.2d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_cl_1d:
; GFX6789: image_sample_cd_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_cd_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_cd_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.1d.v4f32.f32.f32(i32 15, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_cl_2d:
; GFX6789: image_sample_cd_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_cd_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_cd_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.2d.v4f32.f32.f32(i32 15, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_cl_1d:
; GFX6789: image_sample_c_cd_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_cd_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_cd_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.1d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_cl_2d:
; GFX6789: image_sample_c_cd_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_cd_cl v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_cd_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.2d.v4f32.f32.f32(i32 15, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_1d:
; GFX6789: image_sample_l v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_l v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_l_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32 15, float %s, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_2d:
; GFX6789: image_sample_l v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_l v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 15, float %s, float %t, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_1d:
; GFX6789: image_sample_c_l v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_l v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_l_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f32(i32 15, float %zcompare, float %s, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_2d:
; GFX6789: image_sample_c_l v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_l v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.2d.v4f32.f32(i32 15, float %zcompare, float %s, float %t, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_lz_1d:
; GFX6789: image_sample_lz v[0:3], v0, s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_lz v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_lz_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_lz_2d:
; GFX6789: image_sample_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_lz_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_lz_1d:
; GFX6789: image_sample_c_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D ;
define amdgpu_ps <4 x float> @sample_c_lz_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.lz.1d.v4f32.f32(i32 15, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_lz_2d:
; GFX6789: image_sample_c_lz v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_sample_c_lz v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D ;
define amdgpu_ps <4 x float> @sample_c_lz_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.lz.2d.v4f32.f32(i32 15, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_o_2darray_V1:
; GFX6789: image_sample_c_d_o v0, v[0:15], s[0:7], s[8:11] dmask:0x4 da{{$}}
; GFX10: image_sample_c_d_o v0, v[0:15], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D_ARRAY ;
define amdgpu_ps float @sample_c_d_o_2darray_V1(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice) {
main_body:
  %v = call float @llvm.amdgcn.image.sample.c.d.o.2darray.f32.f32.f32(i32 4, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret float %v
}

; GCN-LABEL: {{^}}sample_c_d_o_2darray_V1_tfe:
; GFX6789: image_sample_c_d_o v[9:10], v[0:15], s[0:7], s[8:11] dmask:0x4 tfe da{{$}}
; GFX10: image_sample_c_d_o v[0:1], [v10, v9, v2, v3, v4, v5, v6, v7, v8], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D_ARRAY tfe ;
define amdgpu_ps float @sample_c_d_o_2darray_V1_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice, i32 addrspace(1)* inreg %out) {
main_body:
  %v = call {float,i32} @llvm.amdgcn.image.sample.c.d.o.2darray.f32i32.f32.f32(i32 4, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %v.vec = extractvalue {float, i32} %v, 0
  %v.err = extractvalue {float, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret float %v.vec
}

; GCN-LABEL: {{^}}sample_c_d_o_2darray_V2:
; GFX6789: image_sample_c_d_o v[0:1], v[0:15], s[0:7], s[8:11] dmask:0x6 da{{$}}
; GFX10: image_sample_c_d_o v[0:1], v[0:15], s[0:7], s[8:11] dmask:0x6 dim:SQ_RSRC_IMG_2D_ARRAY ;
define amdgpu_ps <2 x float> @sample_c_d_o_2darray_V2(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice) {
main_body:
  %v = call <2 x float> @llvm.amdgcn.image.sample.c.d.o.2darray.v2f32.f32.f32(i32 6, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <2 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_o_2darray_V2_tfe:
; GFX6789: image_sample_c_d_o v[9:11], v[0:15], s[0:7], s[8:11] dmask:0x6 tfe da{{$}}
; GFX10: image_sample_c_d_o v[0:2], [v11, v10, v9, v3, v4, v5, v6, v7, v8], s[0:7], s[8:11] dmask:0x6 dim:SQ_RSRC_IMG_2D_ARRAY tfe ;
define amdgpu_ps <4 x float> @sample_c_d_o_2darray_V2_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice) {
main_body:
  %v = call {<2 x float>, i32} @llvm.amdgcn.image.sample.c.d.o.2darray.v2f32i32.f32.f32(i32 6, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %v.vec = extractvalue {<2 x float>, i32} %v, 0
  %v.f1 = extractelement <2 x float> %v.vec, i32 0
  %v.f2 = extractelement <2 x float> %v.vec, i32 1
  %v.err = extractvalue {<2 x float>, i32} %v, 1
  %v.errf = bitcast i32 %v.err to float
  %res.0 = insertelement <4 x float> undef, float %v.f1, i32 0
  %res.1 = insertelement <4 x float> %res.0, float %v.f2, i32 1
  %res.2 = insertelement <4 x float> %res.1, float %v.errf, i32 2
  ret <4 x float> %res.2
}

; GCN-LABEL: {{^}}sample_1d_unorm:
; GFX6789: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf unorm{{$}}
; GFX10: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D unorm ;
define amdgpu_ps <4 x float> @sample_1d_unorm(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 1, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_1d_glc:
; GFX6789: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf glc{{$}}
; GFX10: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D glc ;
define amdgpu_ps <4 x float> @sample_1d_glc(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 1)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_1d_slc:
; GFX6789: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf slc{{$}}
; GFX10: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D slc ;
define amdgpu_ps <4 x float> @sample_1d_slc(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 2)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_1d_glc_slc:
; GFX6789: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf glc slc{{$}}
; GFX10: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D glc slc ;
define amdgpu_ps <4 x float> @sample_1d_glc_slc(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 3)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}adjust_writemask_sample_0:
; GCN: image_sample v0, v0, s[0:7], s[8:11] dmask:0x1
define amdgpu_ps float @adjust_writemask_sample_0(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %elt0 = extractelement <4 x float> %r, i32 0
  ret float %elt0
}

; GCN-LABEL: {{^}}adjust_writemask_sample_01
; GCN: image_sample v[0:1], v0, s[0:7], s[8:11] dmask:0x3
define amdgpu_ps <2 x float> @adjust_writemask_sample_01(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %out
}

; GCN-LABEL: {{^}}adjust_writemask_sample_012
; GCN: image_sample v[0:2], v0, s[0:7], s[8:11] dmask:0x7
define amdgpu_ps <3 x float> @adjust_writemask_sample_012(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %out
}

; GCN-LABEL: {{^}}adjust_writemask_sample_12
; GCN: image_sample v[0:1], v0, s[0:7], s[8:11] dmask:0x6
define amdgpu_ps <2 x float> @adjust_writemask_sample_12(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %out
}

; GCN-LABEL: {{^}}adjust_writemask_sample_03
; GCN: image_sample v[0:1], v0, s[0:7], s[8:11] dmask:0x9
define amdgpu_ps <2 x float> @adjust_writemask_sample_03(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <2 x i32> <i32 0, i32 3>
  ret <2 x float> %out
}

; GCN-LABEL: {{^}}adjust_writemask_sample_13
; GCN: image_sample v[0:1], v0, s[0:7], s[8:11] dmask:0xa
define amdgpu_ps <2 x float> @adjust_writemask_sample_13(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <2 x i32> <i32 1, i32 3>
  ret <2 x float> %out
}

; GCN-LABEL: {{^}}adjust_writemask_sample_123
; GCN: image_sample v[0:2], v0, s[0:7], s[8:11] dmask:0xe
define amdgpu_ps <3 x float> @adjust_writemask_sample_123(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <3 x i32> <i32 1, i32 2, i32 3>
  ret <3 x float> %out
}

; GCN-LABEL: {{^}}adjust_writemask_sample_none_enabled
; GCN-NOT: image
define amdgpu_ps <4 x float> @adjust_writemask_sample_none_enabled(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %r
}

; GCN-LABEL: {{^}}adjust_writemask_sample_123_to_12
; GCN: image_sample v[0:1], v0, s[0:7], s[8:11] dmask:0x6
define amdgpu_ps <2 x float> @adjust_writemask_sample_123_to_12(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 14, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %out
}

; GCN-LABEL: {{^}}adjust_writemask_sample_013_to_13
; GCN: image_sample v[0:1], v0, s[0:7], s[8:11] dmask:0xa
define amdgpu_ps <2 x float> @adjust_writemask_sample_013_to_13(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 11, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %out = shufflevector <4 x float> %r, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %out
}

declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.2darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.c.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cl.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cl.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.1d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.2d.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.b.1d.v4f32.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.2d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.2d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.2d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.2d.v4f32.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.1d.v4f32.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.cd.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.1d.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.1d.v4f32.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.2d.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare float @llvm.amdgcn.image.sample.c.d.o.2darray.f32.f32.f32(i32, i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {float, i32} @llvm.amdgcn.image.sample.c.d.o.2darray.f32i32.f32.f32(i32, i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.sample.c.d.o.2darray.v2f32.f32.f32(i32, i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {<2 x float>, i32} @llvm.amdgcn.image.sample.c.d.o.2darray.v2f32i32.f32.f32(i32, i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
