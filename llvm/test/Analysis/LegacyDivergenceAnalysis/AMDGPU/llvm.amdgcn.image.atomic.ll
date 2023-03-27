;RUN: opt -mtriple=amdgcn-mesa-mesa3d -passes='print<divergence>' 2>&1 -disable-output %s | FileCheck %s

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(
define float @image_atomic_swap(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(
define float @image_atomic_add(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.sub.1d.i32.i32(
define float @image_atomic_sub(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.sub.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.smin.1d.i32.i32(
define float @image_atomic_smin(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.smin.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.umin.1d.i32.i32(
define float @image_atomic_umin(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.umin.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.smax.1d.i32.i32(
define float @image_atomic_smax(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.smax.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.umax.1d.i32.i32(
define float @image_atomic_umax(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.umax.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.and.1d.i32.i32(
define float @image_atomic_and(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.and.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.or.1d.i32.i32(
define float @image_atomic_or(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.or.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.xor.1d.i32.i32(
define float @image_atomic_xor(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.xor.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.inc.1d.i32.i32(
define float @image_atomic_inc(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.inc.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.dec.1d.i32.i32(
define float @image_atomic_dec(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.dec.1d.i32.i32(i32 %data, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(
define float @image_atomic_cmpswap(<8 x i32> inreg %rsrc, i32 inreg %addr, i32 inreg %data, i32 inreg %cmp) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32 %data, i32 %cmp, i32 %addr, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.image.atomic.add.2d.i32.i32(
define float @image_atomic_add_2d(<8 x i32> inreg %rsrc, i32 inreg %s, i32 inreg %t, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.image.atomic.add.2d.i32.i32(i32 %data, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
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

attributes #0 = { nounwind }
