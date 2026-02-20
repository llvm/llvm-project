; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_ps void @sample_1d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  call void @llvm.amdgcn.image.sample.1d.nortn.f32(i32 31, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.1d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_2d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  call void @llvm.amdgcn.image.sample.2d.nortn.f32(i32 31, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.2d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_3d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %r) {
main_body:
  call void @llvm.amdgcn.image.sample.3d.nortn.f32(i32 31, float %s, float %t, float %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.3d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_cube_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %face) {
main_body:
  call void @llvm.amdgcn.image.sample.cube.nortn.f32(i32 31, float %s, float %t, float %face, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.cube.nortn.f32
  ret void
}

define amdgpu_ps void @sample_1darray_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %slice) {
main_body:
  call void @llvm.amdgcn.image.sample.1darray.nortn.f32(i32 31, float %s, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.1darray.nortn.f32
  ret void
}

define amdgpu_ps void @sample_2darray_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %slice) {
main_body:
  call void @llvm.amdgcn.image.sample.2darray.nortn.f32(i32 31, float %s, float %t, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.2darray.nortn.f32
  ret void
}

define amdgpu_ps void @sample_b_1d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s) {
main_body:
  call void @llvm.amdgcn.image.sample.b.1d.nortn.f32(i32 31, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.b.1d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_b_2d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t) {
main_body:
  call void @llvm.amdgcn.image.sample.b.2d.nortn.f32(i32 31, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.b.2d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_c_1d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s) {
main_body:
  call void @llvm.amdgcn.image.sample.c.1d.nortn.f32(i32 31, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.c.1d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_c_2d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t) {
main_body:
  call void @llvm.amdgcn.image.sample.c.2d.nortn.f32(i32 31, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.c.2d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_d_1d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dsdv, float %s) {
main_body:
  call void @llvm.amdgcn.image.sample.d.1d.nortn.f32.f32(i32 31, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.d.1d.nortn.f32.f32
  ret void
}

define amdgpu_ps void @sample_d_2d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  call void @llvm.amdgcn.image.sample.d.2d.nortn.f32.f32(i32 31, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.d.2d.nortn.f32.f32
  ret void
}

define amdgpu_ps void @sample_l_1d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %lod) {
main_body:
  call void @llvm.amdgcn.image.sample.l.1d.nortn.f32(i32 31, float %s, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.l.1d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_l_2d_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %lod) {
main_body:
  call void @llvm.amdgcn.image.sample.l.2d.nortn.f32(i32 31, float %s, float %t, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.l.2d.nortn.f32
  ret void
}

define amdgpu_ps void @sample_d_1d_g16_nortn(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dsdv, float %s) {
main_body:
  call void @llvm.amdgcn.image.sample.d.1d.nortn.f16.f32(i32 -1, half %dsdh, half %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.sample.d.1d.nortn.f16.f32
  ret void
}

declare void @llvm.amdgcn.image.sample.1d.nortn.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.2d.nortn.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.3d.nortn.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.cube.nortn.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.1darray.nortn.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.2darray.nortn.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

declare void @llvm.amdgcn.image.sample.b.1d.nortn.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.b.2d.nortn.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

declare void @llvm.amdgcn.image.sample.c.1d.nortn.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.c.2d.nortn.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

declare void @llvm.amdgcn.image.sample.d.1d.f32.nortn.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.d.2d.f32.nortn.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

declare void @llvm.amdgcn.image.sample.l.1d.nortn.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare void @llvm.amdgcn.image.sample.l.2d.nortn.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare void @llvm.amdgcn.image.sample.d.1d.nortn.f16.f32(i32, half, half, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
