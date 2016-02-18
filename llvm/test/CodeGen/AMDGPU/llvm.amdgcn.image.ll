;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}image_load_v4i32:
;CHECK: image_load v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v[0:3], s[0:7]
;CHECK: s_waitcnt vmcnt(0)
define <4 x float> @image_load_v4i32(<8 x i32> inreg %rsrc, <4 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v4i32(<4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret <4 x float> %tex
}

;CHECK-LABEL: {{^}}image_load_v2i32:
;CHECK: image_load v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v[0:1], s[0:7]
;CHECK: s_waitcnt vmcnt(0)
define <4 x float> @image_load_v2i32(<8 x i32> inreg %rsrc, <2 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v2i32(<2 x i32> %c, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret <4 x float> %tex
}

;CHECK-LABEL: {{^}}image_load_i32:
;CHECK: image_load v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v0, s[0:7]
;CHECK: s_waitcnt vmcnt(0)
define <4 x float> @image_load_i32(<8 x i32> inreg %rsrc, i32 %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.i32(i32 %c, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret <4 x float> %tex
}

;CHECK-LABEL: {{^}}image_load_mip:
;CHECK: image_load_mip v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v[0:3], s[0:7]
;CHECK: s_waitcnt vmcnt(0)
define <4 x float> @image_load_mip(<8 x i32> inreg %rsrc, <4 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.mip.v4i32(<4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret <4 x float> %tex
}

;CHECK-LABEL: {{^}}image_load_1:
;CHECK: image_load v0, 1, -1, 0, 0, 0, 0, 0, 0, v[0:3], s[0:7]
;CHECK: s_waitcnt vmcnt(0)
define float @image_load_1(<8 x i32> inreg %rsrc, <4 x i32> %c) #0 {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v4i32(<4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  %elt = extractelement <4 x float> %tex, i32 0
; Only first component used, test that dmask etc. is changed accordingly
  ret float %elt
}

;CHECK-LABEL: {{^}}image_store_v4i32:
;CHECK: image_store v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v[4:7], s[0:7]
define void @image_store_v4i32(<8 x i32> inreg %rsrc, <4 x float> %data, <4 x i32> %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.v4i32(<4 x float> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret void
}

;CHECK-LABEL: {{^}}image_store_v2i32:
;CHECK: image_store v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v[4:5], s[0:7]
define void @image_store_v2i32(<8 x i32> inreg %rsrc, <4 x float> %data, <2 x i32> %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.v2i32(<4 x float> %data, <2 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret void
}

;CHECK-LABEL: {{^}}image_store_i32:
;CHECK: image_store v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v4, s[0:7]
define void @image_store_i32(<8 x i32> inreg %rsrc, <4 x float> %data, i32 %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.i32(<4 x float> %data, i32 %coords, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret void
}

;CHECK-LABEL: {{^}}image_store_mip:
;CHECK: image_store_mip v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v[4:7], s[0:7]
define void @image_store_mip(<8 x i32> inreg %rsrc, <4 x float> %data, <4 x i32> %coords) #0 {
main_body:
  call void @llvm.amdgcn.image.store.mip.v4i32(<4 x float> %data, <4 x i32> %coords, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret void
}

; Ideally, the register allocator would avoid the wait here
;
;CHECK-LABEL: {{^}}image_store_wait:
;CHECK: image_store v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v4, s[0:7]
;CHECK: s_waitcnt vmcnt(0) expcnt(0)
;CHECK: image_load v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v4, s[8:15]
;CHECK: s_waitcnt vmcnt(0)
;CHECK: image_store v[0:3], 15, -1, 0, 0, 0, 0, 0, 0, v4, s[16:23]
define void @image_store_wait(<8 x i32> inreg, <8 x i32> inreg, <8 x i32> inreg, <4 x float>, i32) #0 {
main_body:
  call void @llvm.amdgcn.image.store.i32(<4 x float> %3, i32 %4, <8 x i32> %0, i32 15, i1 0, i1 0, i1 0, i1 0)
  %data = call <4 x float> @llvm.amdgcn.image.load.i32(i32 %4, <8 x i32> %1, i32 15, i1 0, i1 0, i1 0, i1 0)
  call void @llvm.amdgcn.image.store.i32(<4 x float> %data, i32 %4, <8 x i32> %2, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret void
}

declare void @llvm.amdgcn.image.store.i32(<4 x float>, i32, <8 x i32>, i32, i1, i1, i1, i1) #1
declare void @llvm.amdgcn.image.store.v2i32(<4 x float>, <2 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare void @llvm.amdgcn.image.store.v4i32(<4 x float>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare void @llvm.amdgcn.image.store.mip.v4i32(<4 x float>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1

declare <4 x float> @llvm.amdgcn.image.load.i32(i32, <8 x i32>, i32, i1, i1, i1, i1) #2
declare <4 x float> @llvm.amdgcn.image.load.v2i32(<2 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #2
declare <4 x float> @llvm.amdgcn.image.load.v4i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #2
declare <4 x float> @llvm.amdgcn.image.load.mip.v4i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #2

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }
