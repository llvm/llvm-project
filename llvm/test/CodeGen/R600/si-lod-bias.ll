;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; This shader has the potential to generated illegal VGPR to SGPR copies if
; the wrong register class is used for the REG_SEQUENCE instructions.

; CHECK: {{^}}main:
; CHECK: image_sample_b v{{\[[0-9]:[0-9]\]}}, 15, 0, 0, 0, 0, 0, 0, 0, v{{\[[0-9]:[0-9]\]}}

define void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %20 = getelementptr <16 x i8> addrspace(2)* %0, i32 0
  %21 = load <16 x i8> addrspace(2)* %20, !tbaa !1
  %22 = call float @llvm.SI.load.const(<16 x i8> %21, i32 16)
  %23 = getelementptr <32 x i8> addrspace(2)* %2, i32 0
  %24 = load <32 x i8> addrspace(2)* %23, !tbaa !1
  %25 = getelementptr <16 x i8> addrspace(2)* %1, i32 0
  %26 = load <16 x i8> addrspace(2)* %25, !tbaa !1
  %27 = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %3, <2 x i32> %5)
  %28 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %3, <2 x i32> %5)
  %29 = bitcast float %22 to i32
  %30 = bitcast float %27 to i32
  %31 = bitcast float %28 to i32
  %32 = insertelement <4 x i32> undef, i32 %29, i32 0
  %33 = insertelement <4 x i32> %32, i32 %30, i32 1
  %34 = insertelement <4 x i32> %33, i32 %31, i32 2
  %35 = insertelement <4 x i32> %34, i32 undef, i32 3
  %36 = call <4 x float> @llvm.SI.sampleb.v4i32(<4 x i32> %35, <32 x i8> %24, <16 x i8> %26, i32 2)
  %37 = extractelement <4 x float> %36, i32 0
  %38 = extractelement <4 x float> %36, i32 1
  %39 = extractelement <4 x float> %36, i32 2
  %40 = extractelement <4 x float> %36, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %37, float %38, float %39, float %40)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.sampleb.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }

!0 = !{!"const", null}
!1 = !{!0, !0, i64 0, i32 1}
