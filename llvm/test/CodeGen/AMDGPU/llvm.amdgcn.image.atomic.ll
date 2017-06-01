;RUN: llc < %s -march=amdgcn -mcpu=verde -show-mc-encoding -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=SI
;RUN: llc < %s -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=VI

;CHECK-LABEL: {{^}}image_atomic_swap:
;SI: image_atomic_swap v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x3c,0xf0,0x00,0x04,0x00,0x00]
;VI: image_atomic_swap v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x40,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_atomic_swap(<8 x i32> inreg, <4 x i32>, i32) {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.swap.v4i32(i32 %2, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %orig.f = bitcast i32 %orig to float
  ret float %orig.f
}

;CHECK-LABEL: {{^}}image_atomic_swap_v2i32:
;SI: image_atomic_swap v2, v[0:1], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x3c,0xf0,0x00,0x02,0x00,0x00]
;VI: image_atomic_swap v2, v[0:1], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x40,0xf0,0x00,0x02,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_atomic_swap_v2i32(<8 x i32> inreg, <2 x i32>, i32) {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.swap.v2i32(i32 %2, <2 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %orig.f = bitcast i32 %orig to float
  ret float %orig.f
}

;CHECK-LABEL: {{^}}image_atomic_swap_i32:
;SI: image_atomic_swap v1, v0, s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x3c,0xf0,0x00,0x01,0x00,0x00]
;VI: image_atomic_swap v1, v0, s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x40,0xf0,0x00,0x01,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_atomic_swap_i32(<8 x i32> inreg, i32, i32) {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.swap.i32(i32 %2, i32 %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %orig.f = bitcast i32 %orig to float
  ret float %orig.f
}

;CHECK-LABEL: {{^}}image_atomic_cmpswap:
;SI: image_atomic_cmpswap v[4:5], v[0:3], s[0:7] dmask:0x3 unorm glc ; encoding: [0x00,0x33,0x40,0xf0,0x00,0x04,0x00,0x00]
;VI: image_atomic_cmpswap v[4:5], v[0:3], s[0:7] dmask:0x3 unorm glc ; encoding: [0x00,0x33,0x44,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: v_mov_b32_e32 v0, v4
define amdgpu_ps float @image_atomic_cmpswap(<8 x i32> inreg, <4 x i32>, i32, i32) {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.cmpswap.v4i32(i32 %2, i32 %3, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %orig.f = bitcast i32 %orig to float
  ret float %orig.f
}

;CHECK-LABEL: {{^}}image_atomic_add:
;SI: image_atomic_add v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x44,0xf0,0x00,0x04,0x00,0x00]
;VI: image_atomic_add v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x48,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_atomic_add(<8 x i32> inreg, <4 x i32>, i32) {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.add.v4i32(i32 %2, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %orig.f = bitcast i32 %orig to float
  ret float %orig.f
}

;CHECK-LABEL: {{^}}image_atomic_sub:
;SI: image_atomic_sub v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x48,0xf0,0x00,0x04,0x00,0x00]
;VI: image_atomic_sub v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x4c,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_atomic_sub(<8 x i32> inreg, <4 x i32>, i32) {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.sub.v4i32(i32 %2, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %orig.f = bitcast i32 %orig to float
  ret float %orig.f
}

;CHECK-LABEL: {{^}}image_atomic_unchanged:
;CHECK: image_atomic_smin v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x50,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_umin v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x54,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_smax v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x58,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_umax v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x5c,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_and v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x60,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_or v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x64,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_xor v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x68,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_inc v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x6c,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_atomic_dec v4, v[0:3], s[0:7] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x70,0xf0,0x00,0x04,0x00,0x00]
;CHECK: s_waitcnt vmcnt(0)
define amdgpu_ps float @image_atomic_unchanged(<8 x i32> inreg, <4 x i32>, i32) {
main_body:
  %t0 = call i32 @llvm.amdgcn.image.atomic.smin.v4i32(i32 %2, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t1 = call i32 @llvm.amdgcn.image.atomic.umin.v4i32(i32 %t0, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t2 = call i32 @llvm.amdgcn.image.atomic.smax.v4i32(i32 %t1, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t3 = call i32 @llvm.amdgcn.image.atomic.umax.v4i32(i32 %t2, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t4 = call i32 @llvm.amdgcn.image.atomic.and.v4i32(i32 %t3, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t5 = call i32 @llvm.amdgcn.image.atomic.or.v4i32(i32 %t4, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t6 = call i32 @llvm.amdgcn.image.atomic.xor.v4i32(i32 %t5, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t7 = call i32 @llvm.amdgcn.image.atomic.inc.v4i32(i32 %t6, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %t8 = call i32 @llvm.amdgcn.image.atomic.dec.v4i32(i32 %t7, <4 x i32> %1, <8 x i32> %0, i1 0, i1 0, i1 0)
  %out = bitcast i32 %t8 to float
  ret float %out
}

declare i32 @llvm.amdgcn.image.atomic.swap.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.swap.v2i32(i32, <2 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.swap.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0

declare i32 @llvm.amdgcn.image.atomic.cmpswap.v4i32(i32, i32, <4 x i32>, <8 x i32>,i1, i1, i1) #0

declare i32 @llvm.amdgcn.image.atomic.add.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.sub.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.smin.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.umin.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.smax.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.umax.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.and.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.or.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.xor.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.inc.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.dec.v4i32(i32, <4 x i32>, <8 x i32>, i1, i1, i1) #0

attributes #0 = { nounwind }
