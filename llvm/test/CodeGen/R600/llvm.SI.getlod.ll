;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}getlod:
;CHECK: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, 3, 0, 0, -1, 0, 0, 0, 0, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @getlod() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.getlod.i32(i32 undef, <32 x i8> undef, <16 x i8> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r0, float %r1)
  ret void
}

;CHECK-LABEL: {{^}}getlod_v2:
;CHECK: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, 3, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @getlod_v2() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.getlod.v2i32(<2 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r0, float %r1)
  ret void
}

;CHECK-LABEL: {{^}}getlod_v4:
;CHECK: image_get_lod {{v\[[0-9]+:[0-9]+\]}}, 3, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @getlod_v4() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.getlod.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r0, float %r1)
  ret void
}


declare <4 x float> @llvm.SI.getlod.i32(i32, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.getlod.v2i32(<2 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.getlod.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
