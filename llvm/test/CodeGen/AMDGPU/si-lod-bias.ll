;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck %s

; This shader has the potential to generated illegal VGPR to SGPR copies if
; the wrong register class is used for the REG_SEQUENCE instructions.

; CHECK: {{^}}main:
; CHECK: image_sample_b v{{\[[0-9]:[0-9]\]}}, v{{\[[0-9]:[0-9]\]}}, s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf
define amdgpu_ps void @main(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <8 x i32> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 16)
  %tmp22 = getelementptr <8 x i32>, <8 x i32> addrspace(2)* %arg2, i32 0
  %tmp23 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp22, !tbaa !0
  %tmp24 = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg1, i32 0
  %tmp25 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp24, !tbaa !0
  %tmp26 = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %arg3, <2 x i32> %arg5)
  %tmp27 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %arg3, <2 x i32> %arg5)
  %tmp28 = bitcast float %tmp21 to i32
  %tmp29 = bitcast float %tmp26 to i32
  %tmp30 = bitcast float %tmp27 to i32
  %tmp31 = insertelement <4 x i32> undef, i32 %tmp28, i32 0
  %tmp32 = insertelement <4 x i32> %tmp31, i32 %tmp29, i32 1
  %tmp33 = insertelement <4 x i32> %tmp32, i32 %tmp30, i32 2
  %tmp34 = insertelement <4 x i32> %tmp33, i32 undef, i32 3
  %tmp25.bc = bitcast <16 x i8> %tmp25 to <4 x i32>
  %tmp35 = call <4 x float> @llvm.SI.image.sample.b.v4i32(<4 x i32> %tmp34, <8 x i32> %tmp23, <4 x i32> %tmp25.bc, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp36 = extractelement <4 x float> %tmp35, i32 0
  %tmp37 = extractelement <4 x float> %tmp35, i32 1
  %tmp38 = extractelement <4 x float> %tmp35, i32 2
  %tmp39 = extractelement <4 x float> %tmp35, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %tmp36, float %tmp37, float %tmp38, float %tmp39)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #1

declare <4 x float> @llvm.SI.image.sample.b.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1


declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)


attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", !2}
!2 = !{!"tbaa root"}
