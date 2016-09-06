;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}getlod:
;CHECK: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x3 da
define amdgpu_ps void @getlod() {
main_body:
  %r = call <4 x float> @llvm.SI.getlod.i32(i32 undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r0, float %r1)
  ret void
}

;CHECK-LABEL: {{^}}getlod_v2:
;CHECK: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x3 da
define amdgpu_ps void @getlod_v2() {
main_body:
  %r = call <4 x float> @llvm.SI.getlod.v2i32(<2 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r0, float %r1)
  ret void
}

;CHECK-LABEL: {{^}}getlod_v4:
;CHECK: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x3 da
define amdgpu_ps void @getlod_v4() {
main_body:
  %r = call <4 x float> @llvm.SI.getlod.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r0, float %r1)
  ret void
}


declare <4 x float> @llvm.SI.getlod.i32(i32, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #0
declare <4 x float> @llvm.SI.getlod.v2i32(<2 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #0
declare <4 x float> @llvm.SI.getlod.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #0

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind readnone }
