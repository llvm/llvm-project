; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: not --crash llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GFX11-ERR %s

; GFX11-ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.image.gather4

; GCN-LABEL: {{^}}gather4_o_2d:
; GCN: image_gather4_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_o_2d:
; GCN: image_gather4_c_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_c_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_cl_o_2d:
; GCN: image_gather4_cl_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.cl.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_cl_o_2d:
; GCN: image_gather4_c_cl_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_c_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_b_o_2d:
; GCN: image_gather4_b_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_b_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.b.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_b_o_2d:
; GCN: image_gather4_c_b_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_c_b_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.b.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_b_cl_o_2d:
; GCN: image_gather4_b_cl_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_b_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_b_cl_o_2d:
; GCN: image_gather4_c_b_cl_o v[0:3], v[0:5], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_c_b_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %zcompare, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.2d.v4f32.f32.f32(i32 1, i32 %offset, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_l_o_2d:
; GCN: image_gather4_l_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.l.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_l_o_2d:
; GCN: image_gather4_c_l_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_c_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.l.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_lz_o_2d:
; GCN: image_gather4_lz_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_lz_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.lz.o.2d.v4f32.f32(i32 1, i32 %offset, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_lz_o_2d:
; GCN: image_gather4_c_lz_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1{{$}}
define amdgpu_ps <4 x float> @gather4_c_lz_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.2d.v4f32.f32(i32 1, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.gather4.o.2d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.cl.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.cl.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.b.o.2d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.l.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.l.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.lz.o.2d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.lz.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
