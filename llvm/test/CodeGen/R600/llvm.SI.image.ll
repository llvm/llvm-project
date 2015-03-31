;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}image_load:
;CHECK: image_load {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @image_load() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.load.v4i32(<4 x i32> undef, <8 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}image_load_mip:
;CHECK: image_load_mip {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @image_load_mip() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.load.mip.v4i32(<4 x i32> undef, <8 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}getresinfo:
;CHECK: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}
define void @getresinfo() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.getresinfo.i32(i32 undef, <8 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

declare <4 x float> @llvm.SI.image.load.v4i32(<4 x i32>, <8 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.load.mip.v4i32(<4 x i32>, <8 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.getresinfo.i32(i32, <8 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
