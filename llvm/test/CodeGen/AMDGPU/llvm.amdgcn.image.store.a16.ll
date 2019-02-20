; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: {{^}}store.f32.1d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0x1 unorm a16
define amdgpu_ps void @store.f32.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %val, i32 1, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v2f32.1d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0x3 unorm a16
define amdgpu_ps void @store.v2f32.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %val, i32 3, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v3f32.1d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0x7 unorm a16
define amdgpu_ps void @store.v3f32.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %val, i32 7, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v4f32.1d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store.v4f32.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %val, i32 15, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.f32.2d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0x1 unorm a16
define amdgpu_ps void @store.f32.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  call void @llvm.amdgcn.image.store.2d.v4f32.i16(<4 x float> %val, i32 1, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v2f32.2d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0x3 unorm a16
define amdgpu_ps void @store.v2f32.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  call void @llvm.amdgcn.image.store.2d.v4f32.i16(<4 x float> %val, i32 3, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v3f32.2d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0x7 unorm a16
define amdgpu_ps void @store.v3f32.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  call void @llvm.amdgcn.image.store.2d.v4f32.i16(<4 x float> %val, i32 7, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v4f32.2d:
; GCN: image_store v[1:4], v0, s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store.v4f32.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  call void @llvm.amdgcn.image.store.2d.v4f32.i16(<4 x float> %val, i32 15, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.f32.3d:
; GCN: image_store v[2:5], v[0:1], s[0:7] dmask:0x1 unorm a16
define amdgpu_ps void @store.f32.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.3d.v4f32.i16(<4 x float> %val, i32 1, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v2f32.3d:
; GCN: image_store v[2:5], v[0:1], s[0:7] dmask:0x3 unorm a16
define amdgpu_ps void @store.v2f32.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.3d.v4f32.i16(<4 x float> %val, i32 3, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v3f32.3d:
; GCN: image_store v[2:5], v[0:1], s[0:7] dmask:0x7 unorm a16
define amdgpu_ps void @store.v3f32.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.3d.v4f32.i16(<4 x float> %val, i32 7, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}store.v4f32.3d:
; GCN: image_store v[2:5], v[0:1], s[0:7] dmask:0xf unorm a16
define amdgpu_ps void @store.v4f32.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <4 x float> %val) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  call void @llvm.amdgcn.image.store.3d.v4f32.i16(<4 x float> %val, i32 15, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float>, i32, i16, <8 x i32>, i32, i32) #2
declare void @llvm.amdgcn.image.store.2d.v4f32.i16(<4 x float>, i32, i16, i16, <8 x i32>, i32, i32) #2
declare void @llvm.amdgcn.image.store.3d.v4f32.i16(<4 x float>, i32, i16, i16, i16, <8 x i32>, i32, i32) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
