; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}atomic_swap_1d:
; GCN: image_atomic_swap v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_swap_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_1d:
; GCN: image_atomic_add v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_add_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_sub_1d:
; GCN: image_atomic_sub v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_sub_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.sub.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_smin_1d:
; GCN: image_atomic_smin v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_smin_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.smin.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_umin_1d:
; GCN: image_atomic_umin v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_umin_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.umin.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_smax_1d:
; GCN: image_atomic_smax v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_smax_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.smax.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_umax_1d:
; GCN: image_atomic_umax v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_umax_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.umax.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_and_1d:
; GCN: image_atomic_and v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_and_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.and.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_or_1d:
; GCN: image_atomic_or v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_or_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.or.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_xor_1d:
; GCN: image_atomic_xor v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_xor_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.xor.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_inc_1d:
; GCN: image_atomic_inc v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_inc_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.inc.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_dec_1d:
; GCN: image_atomic_dec v0, v1, s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_dec_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.dec.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_cmpswap_1d:
; GCN: image_atomic_cmpswap v[0:1], v2, s[0:7] dmask:0x3 unorm glc{{$}}
define amdgpu_ps float @atomic_cmpswap_1d(<8 x i32> inreg %rsrc, i32 %cmp, i32 %swap, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32 %cmp, i32 %swap, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_2d:
; GCN: image_atomic_add v0, v[1:2], s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_add_2d(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2d.i32.i32(i32 %data, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_3d:
; GCN: image_atomic_add v0, v[1:4], s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_add_3d(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %r) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.3d.i32.i32(i32 %data, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_cube:
; GCN: image_atomic_add v0, v[1:4], s[0:7] dmask:0x1 unorm glc da{{$}}
define amdgpu_ps float @atomic_add_cube(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %face) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.cube.i32.i32(i32 %data, i32 %s, i32 %t, i32 %face, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_1darray:
; GCN: image_atomic_add v0, v[1:2], s[0:7] dmask:0x1 unorm glc da{{$}}
define amdgpu_ps float @atomic_add_1darray(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %slice) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.1darray.i32.i32(i32 %data, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_2darray:
; GCN: image_atomic_add v0, v[1:4], s[0:7] dmask:0x1 unorm glc da{{$}}
define amdgpu_ps float @atomic_add_2darray(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2darray.i32.i32(i32 %data, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_2dmsaa:
; GCN: image_atomic_add v0, v[1:4], s[0:7] dmask:0x1 unorm glc{{$}}
define amdgpu_ps float @atomic_add_2dmsaa(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2dmsaa.i32.i32(i32 %data, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_2darraymsaa:
; GCN: image_atomic_add v0, v[1:4], s[0:7] dmask:0x1 unorm glc da{{$}}
define amdgpu_ps float @atomic_add_2darraymsaa(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2darraymsaa.i32.i32(i32 %data, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

; GCN-LABEL: {{^}}atomic_add_1d_slc:
; GCN: image_atomic_add v0, v1, s[0:7] dmask:0x1 unorm glc slc{{$}}
define amdgpu_ps float @atomic_add_1d_slc(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 2)
  %out = bitcast i32 %v to float
  ret float %out
}

declare i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.sub.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.smin.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.umin.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.smax.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.umax.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.and.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.or.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.xor.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.inc.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.dec.1d.i32.i32(i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #0

declare i32 @llvm.amdgcn.image.atomic.add.2d.i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.3d.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.cube.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.1darray.i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.2darray.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.2dmsaa.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.2darraymsaa.i32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
