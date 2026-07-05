; RUN: opt -S -mtriple=amdgcn-unknown-unknown < %s | FileCheck %s

define amdgpu_ps float @atomic_swap_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps <2 x float> @atomic_swap_1d_i64(<8 x i32> inreg %rsrc, i64 %data, i32 %s) {
main_body:
  %v = call i64 @llvm.amdgcn.image.atomic.swap.1d.i64.i32(i64 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i64 %v to <2 x float>
  ret <2 x float> %out
}

define amdgpu_ps float @atomic_add_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_sub_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.sub.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_smin_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.smin.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_umin_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.umin.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_smax_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.smax.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_umax_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.umax.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_and_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.and.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_or_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.or.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_xor_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.xor.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_inc_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.inc.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_dec_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.dec.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_cmpswap_1d(<8 x i32> inreg %rsrc, i32 %cmp, i32 %swap, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.cmpswap.1d.i32.i32(i32 %cmp, i32 %swap, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps <2 x float> @atomic_cmpswap_1d_64(<8 x i32> inreg %rsrc, i64 %cmp, i64 %swap, i32 %s) {
main_body:
  %v = call i64 @llvm.amdgcn.image.atomic.cmpswap.1d.i64.i32(i64 %cmp, i64 %swap, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i64 %v to <2 x float>
  ret <2 x float> %out
}

define amdgpu_ps float @atomic_add_2d(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2d.i32.i32(i32 %data, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_add_3d(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %r) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.3d.i32.i32(i32 %data, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_add_cube(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %face) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.cube.i32.i32(i32 %data, i32 %s, i32 %t, i32 %face, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_add_1darray(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %slice) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.1darray.i32.i32(i32 %data, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_add_2darray(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2darray.i32.i32(i32 %data, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_add_2dmsaa(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2dmsaa.i32.i32(i32 %data, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

define amdgpu_ps float @atomic_add_2darraymsaa(<8 x i32> inreg %rsrc, i32 %data, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.add.2darraymsaa.i32.i32(i32 %data, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}

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

declare i64 @llvm.amdgcn.image.atomic.swap.1d.i64.i32(i64, i32, <8 x i32>, i32, i32) #0
declare i64 @llvm.amdgcn.image.atomic.cmpswap.1d.i64.i32(i64, i64, i32, <8 x i32>, i32, i32) #0

declare i32 @llvm.amdgcn.image.atomic.add.2d.i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.3d.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.cube.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.1darray.i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.2darray.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.2dmsaa.i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare i32 @llvm.amdgcn.image.atomic.add.2darraymsaa.i32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0

;.
; CHECK: attributes #[[ATTR0:[0-9]+]] = { nocallback nofree nounwind willreturn }
;.