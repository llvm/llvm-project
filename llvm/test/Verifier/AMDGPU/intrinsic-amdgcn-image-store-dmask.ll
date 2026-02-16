; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_ps void @store_1d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.1d.v4f32.i32
  ret void
}

define amdgpu_ps void @store_2d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t) {
main_body:
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.2d.v4f32.i32
  ret void
}

define amdgpu_ps void @store_3d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %r) {
main_body:
  call void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.3d.v4f32.i32
  ret void
}

define amdgpu_ps void @store_cube(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice) {
main_body:
  call void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.cube.v4f32.i32
  ret void
}

define amdgpu_ps void @store_1darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %slice) {
main_body:
  call void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.1darray.v4f32.i32
  ret void
}

define amdgpu_ps void @store_2darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice) {
main_body:
  call void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.2darray.v4f32.i32
  ret void
}

define amdgpu_ps void @store_2dmsaa(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %fragid) {
main_body:
  call void @llvm.amdgcn.image.store.2dmsaa.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.2dmsaa.v4f32.i32
  ret void
}

define amdgpu_ps void @store_2darraymsaa(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  call void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32
  ret void
}

define amdgpu_ps void @store_mip_1d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.mip.1d.v4f32.i32
  ret void
}

define amdgpu_ps void @store_mip_2d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.mip.2d.v4f32.i32
  ret void
}

define amdgpu_ps void @store_mip_3d(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %r, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %r, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.mip.3d.v4f32.i32
  ret void
}

define amdgpu_ps void @store_mip_cube(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.mip.cube.v4f32.i32
  ret void
}

define amdgpu_ps void @store_mip_1darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %slice, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.mip.1darray.v4f32.i32
  ret void
}

define amdgpu_ps void @store_mip_2darray(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  call void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.mip.2darray.v4f32.i32
  ret void
}

define amdgpu_ps void @store_1d_V1(<8 x i32> inreg %rsrc, float %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.f32.i32(float %vdata, i32 31, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.1d.f32.i32
  ret void
}

define amdgpu_ps void @store_1d_V2(<8 x i32> inreg %rsrc, <2 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v2f32.i32(<2 x float> %vdata, i32 31, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.1d.v2f32.i32
  ret void
}

define amdgpu_ps void @store_1d_glc(<8 x i32> inreg %rsrc, <4 x float> %vdata, i32 %s) {
main_body:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %vdata, i32 31, i32 %s, <8 x i32> %rsrc, i32 0, i32 1)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.store.1d.v4f32.i32
  ret void
}

declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2dmsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0

declare void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.1d.f32.i32(float, i32, i32, <8 x i32>, i32, i32) #0
declare void @llvm.amdgcn.image.store.1d.v2f32.i32(<2 x float>, i32, i32, <8 x i32>, i32, i32) #0

attributes #0 = { nounwind }
