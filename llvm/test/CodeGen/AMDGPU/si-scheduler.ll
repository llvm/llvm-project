; FIXME: The si scheduler crashes if when lane mask tracking is enabled, so
; we need to disable this when the si scheduler is being used.
; The only way the subtarget knows that the si machine scheduler is being used
; is to specify -mattr=si-scheduler.  If we just pass --misched=si, the backend
; won't know what scheduler we are using.
; RUN: llc -march=amdgcn -mcpu=SI --misched=si -mattr=si-scheduler < %s | FileCheck %s

; The test checks the "si" machine scheduler pass works correctly.

; CHECK-LABEL: {{^}}main:
; CHECK: s_wqm
; CHECK: s_load_dwordx4
; CHECK: s_load_dwordx8
; CHECK: s_waitcnt lgkmcnt(0)
; CHECK: image_sample
; CHECK: s_waitcnt vmcnt(0)
; CHECK: exp
; CHECK: s_endpgm
define amdgpu_ps void @main([6 x <16 x i8>] addrspace(2)* byval %arg, [17 x <16 x i8>] addrspace(2)* byval %arg1, [17 x <4 x i32>] addrspace(2)* byval %arg2, [34 x <8 x i32>] addrspace(2)* byval %arg3, float inreg %arg4, i32 inreg %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <3 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, i32 %arg19, float %arg20, float %arg21) {
main_body:
  %tmp = bitcast [34 x <8 x i32>] addrspace(2)* %arg3 to <32 x i8> addrspace(2)*
  %tmp22 = load <32 x i8>, <32 x i8> addrspace(2)* %tmp, align 32, !tbaa !0
  %tmp23 = bitcast [17 x <4 x i32>] addrspace(2)* %arg2 to <16 x i8> addrspace(2)*
  %tmp24 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp23, align 16, !tbaa !0
  %i.i = extractelement <2 x i32> %arg11, i32 0
  %j.i = extractelement <2 x i32> %arg11, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 0, i32 0, i32 %arg5) #1
  %p2.i = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 0, i32 0, i32 %arg5) #1
  %i.i1 = extractelement <2 x i32> %arg11, i32 0
  %j.i2 = extractelement <2 x i32> %arg11, i32 1
  %i.f.i3 = bitcast i32 %i.i1 to float
  %j.f.i4 = bitcast i32 %j.i2 to float
  %p1.i5 = call float @llvm.amdgcn.interp.p1(float %i.f.i3, i32 1, i32 0, i32 %arg5) #1
  %p2.i6 = call float @llvm.amdgcn.interp.p2(float %p1.i5, float %j.f.i4, i32 1, i32 0, i32 %arg5) #1
  %tmp27 = bitcast float %p2.i to i32
  %tmp28 = bitcast float %p2.i6 to i32
  %tmp29 = insertelement <2 x i32> undef, i32 %tmp27, i32 0
  %tmp30 = insertelement <2 x i32> %tmp29, i32 %tmp28, i32 1
  %tmp22.bc = bitcast <32 x i8> %tmp22 to <8 x i32>
  %tmp24.bc = bitcast <16 x i8> %tmp24 to <4 x i32>
  %tmp31 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> %tmp30, <8 x i32> %tmp22.bc, <4 x i32> %tmp24.bc, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp32 = extractelement <4 x float> %tmp31, i32 0
  %tmp33 = extractelement <4 x float> %tmp31, i32 1
  %tmp34 = extractelement <4 x float> %tmp31, i32 2
  %tmp35 = extractelement <4 x float> %tmp31, i32 3
  %tmp36 = call i32 @llvm.SI.packf16(float %tmp32, float %tmp33)
  %tmp37 = bitcast i32 %tmp36 to float
  %tmp38 = call i32 @llvm.SI.packf16(float %tmp34, float %tmp35)
  %tmp39 = bitcast i32 %tmp38 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %tmp37, float %tmp39, float %tmp37, float %tmp39)
  ret void
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.SI.packf16(float, float) #0

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #0

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", !2}
!2 = !{!"tbaa root"}
