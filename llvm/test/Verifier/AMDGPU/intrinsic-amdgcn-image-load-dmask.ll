; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_ps void @load_1d(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 31, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.1d.v4f32.i32
  ret void
}

define amdgpu_ps void @load_1d_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 31, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.1d.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_1d_lwe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 -1, i32 %s, <8 x i32> %rsrc, i32 2, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.1d.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_1d_V2_tfe(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call {<2 x float>, i32} @llvm.amdgcn.image.load.1d.v2f32i32.i32(i32 -1, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.1d.sl_v2f32i32s.i32
  ret void
}

define amdgpu_ps void @load_1d_V1_tfe(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call {float, i32} @llvm.amdgcn.image.load.1d.f32i32.i32(i32 3, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: The llvm.amdgcn.image.[load|sample] intrinsic's dmask argument cannot have more active bits than there are elements in the return type
; CHECK-NEXT: @llvm.amdgcn.image.load.1d.sl_f32i32s.i32
  ret void
}

define amdgpu_ps void @load_1d_V2(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <2 x float> @llvm.amdgcn.image.load.1d.v2f32.i32(i32 7, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: The llvm.amdgcn.image.[load|sample] intrinsic's dmask argument cannot have more active bits than there are elements in the return type
; CHECK-NEXT: @llvm.amdgcn.image.load.1d.v2f32.i32
  ret void
}

define amdgpu_ps void @load_1d_V1(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call float @llvm.amdgcn.image.load.1d.f32.i32(i32 3, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: The llvm.amdgcn.image.[load|sample] intrinsic's dmask argument cannot have more active bits than there are elements in the return type
; CHECK-NEXT: @llvm.amdgcn.image.load.1d.f32.i32
  ret void
}

define amdgpu_ps void @load_2d(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 -1, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2d.v4f32.i32
  ret void
}

define amdgpu_ps void @load_2d_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.2d.v4f32i32.i32(i32 31, i32 %s, i32 %t, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2d.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @image_load_mmo(<8 x i32> inreg %rsrc, ptr addrspace(3) %lds, <2 x i32> %c) #0 {
  %c0 = extractelement <2 x i32> %c, i32 0
  %c1 = extractelement <2 x i32> %c, i32 1
  %tex = call float @llvm.amdgcn.image.load.2d.f32.i32(i32 3, i32 %c0, i32 %c1, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: The llvm.amdgcn.image.[load|sample] intrinsic's dmask argument cannot have more active bits than there are elements in the return type
; CHECK-NEXT: @llvm.amdgcn.image.load.2d.f32.i32
  ret void
}

define amdgpu_ps void @load_3d(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %r) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.3d.v4f32.i32
  ret void
}

define amdgpu_ps void @load_3d_tfe_lwe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %r) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.3d.v4f32i32.i32(i32 31, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 3, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.3d.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_cube(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.cube.v4f32.i32
  ret void
}

define amdgpu_ps void @load_cube_lwe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.cube.v4f32i32.i32(i32 31, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 2, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.cube.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_1darray(<8 x i32> inreg %rsrc, i32 %s, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32 31, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.1darray.v4f32.i32
  ret void
}

define amdgpu_ps void @load_1darray_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %slice) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.1darray.v4f32i32.i32(i32 31, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.1darray.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_2darray(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2darray.v4f32.i32
  ret void
}

define amdgpu_ps void @load_2darray_lwe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.2darray.v4f32i32.i32(i32 31, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 2, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2darray.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_2dmsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2dmsaa.v4f32.i32
  ret void
}

define amdgpu_ps void @load_2dmsaa_both(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.2dmsaa.v4f32i32.i32(i32 31, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 3, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2dmsaa.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_2darraymsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32
  ret void
}

define amdgpu_ps void @load_2darraymsaa_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.2darraymsaa.v4f32i32.i32(i32 31, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.2darraymsaa.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_mip_1d(<8 x i32> inreg %rsrc, i32 %s, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32 31, i32 %s, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.1d.v4f32.i32
  ret void
}

define amdgpu_ps void @load_mip_1d_lwe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %mip) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.mip.1d.v4f32i32.i32(i32 31, i32 %s, i32 %mip, <8 x i32> %rsrc, i32 2, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.1d.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_mip_2d(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.2d.v4f32.i32
  ret void
}

define amdgpu_ps void @load_mip_2d_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %mip) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.load.mip.2d.v4f32i32.i32(i32 31, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.2d.sl_v4f32i32s.i32
  ret void
}

define amdgpu_ps void @load_mip_2d_tfe_nouse_V2(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %mip) {
main_body:
  %v = call {<2 x float>, i32} @llvm.amdgcn.image.load.mip.2d.v2f32i32.i32(i32 7, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: The llvm.amdgcn.image.[load|sample] intrinsic's dmask argument cannot have more active bits than there are elements in the return type
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.2d.sl_v2f32i32s.i32
  ret void
}

define amdgpu_ps void @load_mip_2d_tfe_nouse_V1(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %mip) {
main_body:
  %v = call {float, i32} @llvm.amdgcn.image.load.mip.2d.f32i32.i32(i32 3, i32 %s, i32 %t, i32 %mip, <8 x i32> %rsrc, i32 1, i32 0)
; CHECK: The llvm.amdgcn.image.[load|sample] intrinsic's dmask argument cannot have more active bits than there are elements in the return type
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.2d.sl_f32i32s.i32
  ret void
}

define amdgpu_ps void @load_mip_3d(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %r, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %r, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.3d.v4f32.i32
  ret void
}

define amdgpu_ps void @load_mip_cube(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.cube.v4f32.i32
  ret void
}

define amdgpu_ps void @load_mip_1darray(<8 x i32> inreg %rsrc, i32 %s, i32 %slice, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32 31, i32 %s, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.1darray.v4f32.i32
  ret void
}

define amdgpu_ps void @load_mip_2darray(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %mip) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32 31, i32 %s, i32 %t, i32 %slice, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
; CHECK: DMask is a 4 bit value and therefore at most 4 least significant bits of an llvm.amdgcn.image.* intrinsic's dmask argument can be set
; CHECK-NEXT: @llvm.amdgcn.image.load.mip.2darray.v4f32.i32
  ret void
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare {float, i32} @llvm.amdgcn.image.load.1d.f32i32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare {<2 x float>, i32} @llvm.amdgcn.image.load.1d.v2f32i32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.2d.v4f32i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.3d.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.cube.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.1darray.v4f32i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.2darray.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.2dmsaa.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.2darraymsaa.v4f32i32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.mip.1d.v4f32i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>, i32} @llvm.amdgcn.image.load.mip.2d.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<2 x float>, i32} @llvm.amdgcn.image.load.mip.2d.v2f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {float, i32} @llvm.amdgcn.image.load.mip.2d.f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare float @llvm.amdgcn.image.load.1d.f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare float @llvm.amdgcn.image.load.2d.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.load.1d.v2f32.i32(i32, i32, <8 x i32>, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
