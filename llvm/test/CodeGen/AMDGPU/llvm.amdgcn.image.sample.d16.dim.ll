; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=UNPACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx810 -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX81 %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}image_sample_2d_f16:
; GCN: image_sample v0, v[0:1], s[0:7], s[8:11] dmask:0x1 d16{{$}}
define amdgpu_ps half @image_sample_2d_f16(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  %tex = call half @llvm.amdgcn.image.sample.2d.f16.f32(i32 1, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  ret half %tex
}

; GCN-LABEL: {{^}}image_sample_2d_f16_tfe:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; PACKED: image_sample v[2:3], v[0:1], s[0:7], s[8:11] dmask:0x1 tfe d16{{$}}
; UNPACKED: image_sample v[2:3], v[0:1], s[0:7], s[8:11] dmask:0x1 tfe d16{{$}}
define amdgpu_ps half @image_sample_2d_f16_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, i32 addrspace(1)* inreg %out) {
main_body:
  %tex = call {half,i32} @llvm.amdgcn.image.sample.2d.f16i32.f32(i32 1, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 0)
  %tex.vec = extractvalue {half, i32} %tex, 0
  %tex.err = extractvalue {half, i32} %tex, 1
  store i32 %tex.err, i32 addrspace(1)* %out, align 4
  ret half %tex.vec
}

; GCN-LABEL: {{^}}image_sample_c_d_1d_v2f16:
; UNPACKED: image_sample_c_d v[0:1], v[0:3], s[0:7], s[8:11] dmask:0x3 d16{{$}}
; PACKED: image_sample_c_d v0, v[0:3], s[0:7], s[8:11] dmask:0x3 d16{{$}}
define amdgpu_ps float @image_sample_c_d_1d_v2f16(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dsdv, float %s) {
main_body:
  %tex = call <2 x half> @llvm.amdgcn.image.sample.c.d.1d.v2f16.f32.f32(i32 3, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  %r = bitcast <2 x half> %tex to float
  ret float %r
}

; GCN-LABEL: {{^}}image_sample_c_d_1d_v2f16_tfe:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; UNPACKED: image_sample_c_d v[{{[0-9]+:[0-9]+}}], v[0:3], s[0:7], s[8:11] dmask:0x3 tfe d16{{$}}
; PACKED: image_sample_c_d v[{{[0-9]+:[0-9]+}}], v[0:3], s[0:7], s[8:11] dmask:0x3 tfe d16{{$}}
define amdgpu_ps <2 x float> @image_sample_c_d_1d_v2f16_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %dsdh, float %dsdv, float %s) {
main_body:
  %tex = call {<2 x half>,i32} @llvm.amdgcn.image.sample.c.d.1d.v2f16i32.f32.f32(i32 3, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 0)
  %tex.vec = extractvalue {<2 x half>, i32} %tex, 0
  %tex.err = extractvalue {<2 x half>, i32} %tex, 1
  %tex.vecf = bitcast <2 x half> %tex.vec to float
  %r.0 = insertelement <2 x float> undef, float %tex.vecf, i32 0
  %tex.errf = bitcast i32 %tex.err to float
  %r = insertelement <2 x float> %r.0, float %tex.errf, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}image_sample_b_2d_v4f16:
; UNPACKED: image_sample_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf d16{{$}}
; PACKED: image_sample_b v[0:1], v[0:3], s[0:7], s[8:11] dmask:0xf d16{{$}}
define amdgpu_ps <2 x float> @image_sample_b_2d_v4f16(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.sample.b.2d.v4f16.f32.f32(i32 15, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  %r = bitcast <4 x half> %tex to <2 x float>
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}image_sample_b_2d_v4f16_tfe:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; UNPACKED: image_sample_b v[{{[0-9]+:[0-9]+}}], v[0:3], s[0:7], s[8:11] dmask:0xf tfe d16{{$}}
; PACKED: image_sample_b v[{{[0-9]+:[0-9]+}}], v[0:3], s[0:7], s[8:11] dmask:0xf tfe d16{{$}}
define amdgpu_ps <4 x float> @image_sample_b_2d_v4f16_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t) {
main_body:
  %tex = call {<4 x half>,i32} @llvm.amdgcn.image.sample.b.2d.v4f16i32.f32.f32(i32 15, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 0)
  %tex.vec = extractvalue {<4 x half>, i32} %tex, 0
  %tex.err = extractvalue {<4 x half>, i32} %tex, 1
  %tex.vecf = bitcast <4 x half> %tex.vec to <2 x float>
  %tex.vecf.0 = extractelement <2 x float> %tex.vecf, i32 0
  %tex.vecf.1 = extractelement <2 x float> %tex.vecf, i32 1
  %r.0 = insertelement <4 x float> undef, float %tex.vecf.0, i32 0
  %r.1 = insertelement <4 x float> %r.0, float %tex.vecf.1, i32 1
  %tex.errf = bitcast i32 %tex.err to float
  %r = insertelement <4 x float> %r.1, float %tex.errf, i32 2
  ret <4 x float> %r
}

declare half @llvm.amdgcn.image.sample.2d.f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {half,i32} @llvm.amdgcn.image.sample.2d.f16i32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x half> @llvm.amdgcn.image.sample.2d.v4f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {<2 x half>,i32} @llvm.amdgcn.image.sample.2d.v2f16i32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <2 x half> @llvm.amdgcn.image.sample.c.d.1d.v2f16.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {<2 x half>,i32} @llvm.amdgcn.image.sample.c.d.1d.v2f16i32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x half> @llvm.amdgcn.image.sample.b.2d.v4f16.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare {<4 x half>,i32} @llvm.amdgcn.image.sample.b.2d.v4f16i32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
