; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=UNPACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx810 -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX81 %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PACKED -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}image_load_f16:
; GCN: image_load v0, v[0:1], s[0:7] dmask:0x1 unorm d16{{$}}
define amdgpu_ps half @image_load_f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %tex = call half @llvm.amdgcn.image.load.2d.f16.i32(i32 1, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret half %tex
}

; GCN-LABEL: {{^}}image_load_v2f16:
; UNPACKED: image_load v[0:1], v[0:1], s[0:7] dmask:0x3 unorm d16{{$}}
; PACKED: image_load v0, v[0:1], s[0:7] dmask:0x3 unorm d16{{$}}
define amdgpu_ps float @image_load_v2f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %tex = call <2 x half> @llvm.amdgcn.image.load.2d.v2f16.i32(i32 3, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast <2 x half> %tex to float
  ret float %r
}

; GCN-LABEL: {{^}}image_load_v4f16:
; UNPACKED: image_load v[0:3], v[0:1], s[0:7] dmask:0xf unorm d16{{$}}
; PACKED: image_load v[0:1], v[0:1], s[0:7] dmask:0xf unorm d16{{$}}
define amdgpu_ps <2 x float> @image_load_v4f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast <4 x half> %tex to <2 x float>
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}image_load_mip_v4f16:
; UNPACKED: image_load_mip v[0:3], v[0:3], s[0:7] dmask:0xf unorm d16{{$}}
; PACKED: image_load_mip v[0:1], v[0:3], s[0:7] dmask:0xf unorm d16{{$}}
define amdgpu_ps <2 x float> @image_load_mip_v4f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %mip) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32 15, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast <4 x half> %tex to <2 x float>
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}image_load_3d_v2f16:
; UNPACKED: image_load v[0:1], v[0:3], s[0:7] dmask:0x3 unorm d16{{$}}
; PACKED: image_load v0, v[0:3], s[0:7] dmask:0x3 unorm d16{{$}}
define amdgpu_ps float @image_load_3d_v2f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %r) {
main_body:
  %tex = call <2 x half> @llvm.amdgcn.image.load.3d.v2f16.i32(i32 3, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
  %x = bitcast <2 x half> %tex to float
  ret float %x
}

; GCN-LABEL: {{^}}image_store_f16
; GCN: image_store v2, v[0:1], s[0:7] dmask:0x1 unorm d16{{$}}
define amdgpu_ps void @image_store_f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, half %data) {
main_body:
  call void @llvm.amdgcn.image.store.2d.f16.i32(half %data, i32 1, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}image_store_v2f16
; UNPACKED: v_lshrrev_b32_e32
; UNPACKED: v_and_b32_e32
; UNPACKED: image_store v[{{[0-9:]+}}], v[0:1], s[0:7] dmask:0x3 unorm d16{{$}}
; PACKED: image_store v2, v[0:1], s[0:7] dmask:0x3 unorm d16{{$}}
define amdgpu_ps void @image_store_v2f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, float %in) {
main_body:
  %data = bitcast float %in to <2 x half>
  call void @llvm.amdgcn.image.store.2d.v2f16.i32(<2 x half> %data, i32 3, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}image_store_v4f16
; UNPACKED: v_lshrrev_b32_e32
; UNPACKED: v_and_b32_e32
; UNPACKED: v_lshrrev_b32_e32
; UNPACKED: v_and_b32_e32
; UNPACKED: image_store v[{{[0-9:]+}}], v[0:1], s[0:7] dmask:0xf unorm d16{{$}}
; PACKED: image_store v[2:3], v[0:1], s[0:7] dmask:0xf unorm d16{{$}}
define amdgpu_ps void @image_store_v4f16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, <2 x float> %in) {
main_body:
  %data = bitcast <2 x float> %in to <4 x half>
  call void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half> %data, i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}image_store_mip_1d_v4f16
; UNPACKED: v_lshrrev_b32_e32
; UNPACKED: v_and_b32_e32
; UNPACKED: v_lshrrev_b32_e32
; UNPACKED: v_and_b32_e32
; UNPACKED: image_store_mip v[{{[0-9:]+}}], v[0:1], s[0:7] dmask:0xf unorm d16{{$}}
; PACKED: image_store_mip v[2:3], v[0:1], s[0:7] dmask:0xf unorm d16{{$}}
define amdgpu_ps void @image_store_mip_1d_v4f16(<8 x i32> inreg %rsrc, i32 %s, i32 %mip, <2 x float> %in) {
main_body:
  %data = bitcast <2 x float> %in to <4 x half>
  call void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half> %data, i32 15, i32 %s, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

declare half @llvm.amdgcn.image.load.2d.f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <2 x half> @llvm.amdgcn.image.load.2d.v2f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <2 x half> @llvm.amdgcn.image.load.3d.v2f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare void @llvm.amdgcn.image.store.2d.f16.i32(half, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2d.v2f16.i32(<2 x half>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.3d.v2f16.i32(<2 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
