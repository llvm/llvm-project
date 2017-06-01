;RUN: llc < %s -march=amdgcn -mcpu=verde | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga | FileCheck %s

; CHECK-LABEL: {{^}}v1:
; CHECK: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xd
define amdgpu_ps void @v1(i32 %a1) {
entry:
  %0 = insertelement <1 x i32> undef, i32 %a1, i32 0
  %1 = call <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32> %0, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %2 = extractelement <4 x float> %1, i32 0
  %3 = extractelement <4 x float> %1, i32 2
  %4 = extractelement <4 x float> %1, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %2, float %3, float %4, float %4)
  ret void
}

; CHECK-LABEL: {{^}}v2:
; CHECK: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xb
define amdgpu_ps void @v2(i32 %a1) {
entry:
  %0 = insertelement <1 x i32> undef, i32 %a1, i32 0
  %1 = call <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32> %0, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %2 = extractelement <4 x float> %1, i32 0
  %3 = extractelement <4 x float> %1, i32 1
  %4 = extractelement <4 x float> %1, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %2, float %3, float %4, float %4)
  ret void
}

; CHECK-LABEL: {{^}}v3:
; CHECK: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xe
define amdgpu_ps void @v3(i32 %a1) {
entry:
  %0 = insertelement <1 x i32> undef, i32 %a1, i32 0
  %1 = call <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32> %0, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %2 = extractelement <4 x float> %1, i32 1
  %3 = extractelement <4 x float> %1, i32 2
  %4 = extractelement <4 x float> %1, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %2, float %3, float %4, float %4)
  ret void
}

; CHECK-LABEL: {{^}}v4:
; CHECK: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x7
define amdgpu_ps void @v4(i32 %a1) {
entry:
  %0 = insertelement <1 x i32> undef, i32 %a1, i32 0
  %1 = call <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32> %0, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %2 = extractelement <4 x float> %1, i32 0
  %3 = extractelement <4 x float> %1, i32 1
  %4 = extractelement <4 x float> %1, i32 2
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %2, float %3, float %4, float %4)
  ret void
}

; CHECK-LABEL: {{^}}v5:
; CHECK: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0xa
define amdgpu_ps void @v5(i32 %a1) {
entry:
  %0 = insertelement <1 x i32> undef, i32 %a1, i32 0
  %1 = call <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32> %0, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %2 = extractelement <4 x float> %1, i32 1
  %3 = extractelement <4 x float> %1, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %2, float %3, float %3, float %3)
  ret void
}

; CHECK-LABEL: {{^}}v6:
; CHECK: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x6
define amdgpu_ps void @v6(i32 %a1) {
entry:
  %0 = insertelement <1 x i32> undef, i32 %a1, i32 0
  %1 = call <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32> %0, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %2 = extractelement <4 x float> %1, i32 1
  %3 = extractelement <4 x float> %1, i32 2
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %2, float %3, float %3, float %3)
  ret void
}

; CHECK-LABEL: {{^}}v7:
; CHECK: image_sample {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x9
define amdgpu_ps void @v7(i32 %a1) {
entry:
  %0 = insertelement <1 x i32> undef, i32 %a1, i32 0
  %1 = call <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32> %0, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %2 = extractelement <4 x float> %1, i32 0
  %3 = extractelement <4 x float> %1, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %2, float %3, float %3, float %3)
  ret void
}

declare <4 x float> @llvm.SI.image.sample.v1i32(<1 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
