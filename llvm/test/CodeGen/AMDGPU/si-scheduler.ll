; FIXME: The si scheduler crashes if when lane mask tracking is enabled, so
; we need to disable this when the si scheduler is being used.
; The only way the subtarget knows that the si machine scheduler is being used
; is to specify -mattr=si-scheduler.  If we just pass --misched=si, the backend
; won't know what scheduler we are using.
; RUN: llc -march=amdgcn --misched=si -mattr=si-scheduler < %s | FileCheck %s

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
define amdgpu_ps void @main([6 x <16 x i8>] addrspace(4)* byval %arg, [17 x <16 x i8>] addrspace(4)* byval %arg1, [17 x <4 x i32>] addrspace(4)* byval %arg2, [34 x <8 x i32>] addrspace(4)* byval %arg3, float inreg %arg4, i32 inreg %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <3 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, i32 %arg19, float %arg20, float %arg21) #0 {
main_body:
  %tmp = bitcast [34 x <8 x i32>] addrspace(4)* %arg3 to <32 x i8> addrspace(4)*
  %tmp22 = load <32 x i8>, <32 x i8> addrspace(4)* %tmp, align 32, !tbaa !0
  %tmp23 = bitcast [17 x <4 x i32>] addrspace(4)* %arg2 to <16 x i8> addrspace(4)*
  %tmp24 = load <16 x i8>, <16 x i8> addrspace(4)* %tmp23, align 16, !tbaa !0
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
  %tmp30.bc = bitcast <2 x i32> %tmp30 to <2 x float>
  %tmp31 = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float> %tmp30.bc, <8 x i32> %tmp22.bc, <4 x i32> %tmp24.bc, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)

  %tmp32 = extractelement <4 x float> %tmp31, i32 0
  %tmp33 = extractelement <4 x float> %tmp31, i32 1
  %tmp34 = extractelement <4 x float> %tmp31, i32 2
  %tmp35 = extractelement <4 x float> %tmp31, i32 3
  %tmp36 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp32, float %tmp33)
  %tmp38 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp34, float %tmp35)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %tmp36, <2 x half> %tmp38, i1 true, i1 false) #0
  ret void
}

declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #1
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #1
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #0
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1
declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", !2}
!2 = !{!"tbaa root"}
