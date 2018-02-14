; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; This shader has the potential to generated illegal VGPR to SGPR copies if
; the wrong register class is used for the REG_SEQUENCE instructions.

; GCN-LABEL: {{^}}main:
; GCN: image_sample_b v{{\[[0-9]:[0-9]\]}}, v{{\[[0-9]:[0-9]\]}}, s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0xf
define amdgpu_ps void @main(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <8 x i32> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp, !tbaa !0
  %tmp21 = call float @llvm.SI.load.const.v4i32(<4 x i32> %tmp20, i32 16)
  %tmp22 = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %arg2, i32 0
  %tmp23 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp22, !tbaa !0
  %tmp24 = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg1, i32 0
  %tmp25 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp24, !tbaa !0
  %i.i = extractelement <2 x i32> %arg5, i32 0
  %j.i = extractelement <2 x i32> %arg5, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 0, i32 0, i32 %arg3) #0
  %p2.i = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 0, i32 0, i32 %arg3) #0
  %i.i1 = extractelement <2 x i32> %arg5, i32 0
  %j.i2 = extractelement <2 x i32> %arg5, i32 1
  %i.f.i3 = bitcast i32 %i.i1 to float
  %j.f.i4 = bitcast i32 %j.i2 to float
  %p1.i5 = call float @llvm.amdgcn.interp.p1(float %i.f.i3, i32 1, i32 0, i32 %arg3) #0
  %p2.i6 = call float @llvm.amdgcn.interp.p2(float %p1.i5, float %j.f.i4, i32 1, i32 0, i32 %arg3) #0
  %tmp28 = bitcast float %tmp21 to i32
  %tmp29 = bitcast float %p2.i to i32
  %tmp30 = bitcast float %p2.i6 to i32
  %tmp31 = insertelement <4 x i32> undef, i32 %tmp28, i32 0
  %tmp32 = insertelement <4 x i32> %tmp31, i32 %tmp29, i32 1
  %tmp33 = insertelement <4 x i32> %tmp32, i32 %tmp30, i32 2
  %tmp34 = insertelement <4 x i32> %tmp33, i32 undef, i32 3
  %tmp34.bc = bitcast <4 x i32> %tmp34 to <4 x float>
  %tmp35 = call <4 x float> @llvm.amdgcn.image.sample.b.v4f32.v4f32.v8i32(<4 x float> %tmp34.bc, <8 x i32> %tmp23, <4 x i32> %tmp25, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp36 = extractelement <4 x float> %tmp35, i32 0
  %tmp37 = extractelement <4 x float> %tmp35, i32 1
  %tmp38 = extractelement <4 x float> %tmp35, i32 2
  %tmp39 = extractelement <4 x float> %tmp35, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp36, float %tmp37, float %tmp38, float %tmp39, i1 true, i1 true) #0
  ret void
}

declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #1
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #1
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.b.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #2
declare float @llvm.SI.load.const.v4i32(<4 x i32>, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", !2}
!2 = !{!"tbaa root"}
