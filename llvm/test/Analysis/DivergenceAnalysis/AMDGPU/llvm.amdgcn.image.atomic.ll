;RUN: opt -mtriple=amdgcn-mesa-mesa3d -analyze -divergence %s | FileCheck %s

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.swap.i32(
define float @image_atomic_swap(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.swap.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.add.i32(
define float @image_atomic_add(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.add.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.sub.i32(
define float @image_atomic_sub(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.sub.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.smin.i32(
define float @image_atomic_smin(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.smin.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.umin.i32(
define float @image_atomic_umin(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.umin.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.smax.i32(
define float @image_atomic_smax(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.smax.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.umax.i32(
define float @image_atomic_umax(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.umax.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.and.i32(
define float @image_atomic_and(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.and.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.or.i32(
define float @image_atomic_or(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.or.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.xor.i32(
define float @image_atomic_xor(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.xor.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.inc.i32(
define float @image_atomic_inc(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.inc.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.dec.i32(
define float @image_atomic_dec(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.dec.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.cmpswap.i32(
define float @image_atomic_cmpswap(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data, i32 inreg %cmp) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.cmpswap.i32(i32 %data, i32 %cmp, i32 %addr, <8 x i32> %rsrc, i1 0, i1 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

declare i32 @llvm.amdgcn.image.atomic.swap.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.add.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.sub.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.smin.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.umin.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.smax.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.umax.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.and.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.or.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.xor.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.inc.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.dec.i32(i32, i32, <8 x i32>, i1, i1, i1) #0
declare i32 @llvm.amdgcn.image.atomic.cmpswap.i32(i32, i32, i32, <8 x i32>,i1, i1, i1) #0

attributes #0 = { nounwind }
