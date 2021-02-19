;RUN: opt -mtriple=amdgcn-mesa-mesa3d -amdgpu-use-legacy-divergence-analysis -enable-new-pm=0 -analyze -divergence %s | FileCheck %s

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(
define float @buffer_atomic_swap(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.add.i32(
define float @buffer_atomic_add(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.add.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.sub.i32(
define float @buffer_atomic_sub(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.sub.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.smin.i32(
define float @buffer_atomic_smin(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.smin.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.umin.i32(
define float @buffer_atomic_umin(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.umin.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.smax.i32(
define float @buffer_atomic_smax(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.smax.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.umax.i32(
define float @buffer_atomic_umax(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.umax.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.and.i32(
define float @buffer_atomic_and(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.and.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.or.i32(
define float @buffer_atomic_or(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.or.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.xor.i32(
define float @buffer_atomic_xor(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.xor.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

;CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(
define float @buffer_atomic_cmpswap(<4 x i32> inreg %rsrc, i32 inreg %data, i32 inreg %cmp) #0 {
main_body:
  %orig = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %data, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

declare i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.add.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.sub.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.smin.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.umin.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.smax.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.umax.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.and.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.or.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.xor.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32, i32, <4 x i32>, i32, i32, i1) #0

attributes #0 = { nounwind }
