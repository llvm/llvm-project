; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=UNPACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx810 -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX81 %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX9 %s


; GCN-LABEL: {{^}}image_sample_f16:
; GCN: image_sample v[[HALF:[0-9]+]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x1 d16

; UNPACKED: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]]

; GFX81: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]]

; GFX9: global_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]], off
define amdgpu_kernel void @image_sample_f16(<4 x float> %coords, <8 x i32> inreg %rsrc, <4 x i32> inreg %sample, half addrspace(1)* %out) {
main_body:
  %tex = call half @llvm.amdgcn.image.sample.f16.v4f32.v8i32(<4 x float> %coords, <8 x i32> %rsrc, <4 x i32> %sample, i32 1, i1 0, i1 0, i1 0, i1 0, i1 0)
  store half %tex, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}image_sample_v2f16:
; UNPACKED: image_sample v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x3 d16
; UNPACKED: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HI]]

; PACKED: image_sample v[[DATA:[0-9]+]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x3 d16

; GFX81: v_lshrrev_b32_e32 v[[HI:[0-9]+]], 16, v[[DATA]]
; GFX81: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HI]]

; GFX9: global_store_short_d16_hi v[{{[0-9]+:[0-9]+}}], v[[DATA]], off
define amdgpu_kernel void @image_sample_v2f16(<4 x float> %coords, <8 x i32> inreg %rsrc, <4 x i32> inreg %sample, half addrspace(1)* %out) {
main_body:
  %tex = call <2 x half> @llvm.amdgcn.image.sample.v2f16.v4f32.v8i32(<4 x float> %coords, <8 x i32> %rsrc, <4 x i32> %sample, i32 3, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt = extractelement <2 x half> %tex, i32 1
  store half %elt, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}image_sample_v4f16:
; UNPACKED: image_sample v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; UNPACKED: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HI]]

; PACKED: image_sample v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; PACKED: v_lshrrev_b32_e32 v[[HALF:[0-9]+]], 16, v[[HI]]

; GFX81: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]]

; GFX9: global_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]], off
define amdgpu_kernel void @image_sample_v4f16(<4 x float> %coords, <8 x i32> inreg %rsrc, <4 x i32> inreg %sample, half addrspace(1)* %out) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.sample.v4f16.v4f32.v8i32(<4 x float> %coords, <8 x i32> %rsrc, <4 x i32> %sample, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt = extractelement <4 x half> %tex, i32 3
  store half %elt, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}image_sample_cl_v4f16:
; UNPACKED: image_sample_cl v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; UNPACKED: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HI]]

; PACKED: image_sample_cl v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; PACKED: v_lshrrev_b32_e32 v[[HALF:[0-9]+]], 16, v[[HI]]

; GFX81: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]]

; GFX9: global_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]], off
define amdgpu_kernel void @image_sample_cl_v4f16(<4 x float> %coords, <8 x i32> inreg %rsrc, <4 x i32> inreg %sample, half addrspace(1)* %out) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.sample.cl.v4f16.v4f32.v8i32(<4 x float> %coords, <8 x i32> %rsrc, <4 x i32> %sample, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt = extractelement <4 x half> %tex, i32 3
  store half %elt, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}image_sample_c_v4f16:
; UNPACKED: image_sample_c v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; UNPACKED: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HI]]

; PACKED: image_sample_c v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; PACKED: v_lshrrev_b32_e32 v[[HALF:[0-9]+]], 16, v[[HI]]

; GFX81: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]]

; GFX9: global_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]], off
define amdgpu_kernel void @image_sample_c_v4f16(<4 x float> %coords, <8 x i32> inreg %rsrc, <4 x i32> inreg %sample, half addrspace(1)* %out) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.sample.c.v4f16.v4f32.v8i32(<4 x float> %coords, <8 x i32> %rsrc, <4 x i32> %sample, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt = extractelement <4 x half> %tex, i32 3
  store half %elt, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}image_sample_o_v4f16:
; UNPACKED: image_sample_o v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; UNPACKED: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HI]]

; PACKED: image_sample_o v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; PACKED: v_lshrrev_b32_e32 v[[HALF:[0-9]+]], 16, v[[HI]]

; GFX81: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]]
; GFX9: global_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]], off
define amdgpu_kernel void @image_sample_o_v4f16(<4 x float> %coords, <8 x i32> inreg %rsrc, <4 x i32> inreg %sample, half addrspace(1)* %out) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.sample.o.v4f16.v4f32.v8i32(<4 x float> %coords, <8 x i32> %rsrc, <4 x i32> %sample, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt = extractelement <4 x half> %tex, i32 3
  store half %elt, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}image_sample_c_o_v4f16:
; UNPACKED: image_sample_c_o v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; UNPACKED: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HI]]

; PACKED: image_sample_c_o v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf d16
; PACKED: v_lshrrev_b32_e32 v[[HALF:[0-9]+]], 16, v[[HI]]

; GFX81: flat_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]]
; GFX9: global_store_short v[{{[0-9]+:[0-9]+}}], v[[HALF]], off
define amdgpu_kernel void @image_sample_c_o_v4f16(<4 x float> %coords, <8 x i32> inreg %rsrc, <4 x i32> inreg %sample, half addrspace(1)* %out) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.sample.c.o.v4f16.v4f32.v8i32(<4 x float> %coords, <8 x i32> %rsrc, <4 x i32> %sample, i32 15, i1 0, i1 0, i1 0, i1 0, i1 0)
  %elt = extractelement <4 x half> %tex, i32 3
  store half %elt, half addrspace(1)* %out
  ret void
}

declare half @llvm.amdgcn.image.sample.f16.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1)
declare <2 x half> @llvm.amdgcn.image.sample.v2f16.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1)
declare <4 x half> @llvm.amdgcn.image.sample.v4f16.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1)


declare <4 x half> @llvm.amdgcn.image.sample.cl.v4f16.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1)
declare <4 x half> @llvm.amdgcn.image.sample.c.v4f16.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1)
declare <4 x half> @llvm.amdgcn.image.sample.o.v4f16.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1)
declare <4 x half> @llvm.amdgcn.image.sample.c.o.v4f16.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1)
