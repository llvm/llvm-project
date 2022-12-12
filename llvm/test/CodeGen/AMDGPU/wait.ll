; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=DEFAULT
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=DEFAULT
; RUN: llc -march=amdgcn --misched=ilpmax -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=ILPMAX
; RUN: llc -march=amdgcn --misched=ilpmax -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=ILPMAX
; The ilpmax scheduler is used for the second test to get the ordering we want for the test.

; DEFAULT-LABEL: {{^}}main:
; DEFAULT: s_load_dwordx8
; DEFAULT: s_waitcnt lgkmcnt(0)
; DEFAULT: buffer_load_format_xyzw
; DEFAULT: buffer_load_format_xyzw
; DEFAULT-DAG: s_waitcnt vmcnt(0)
; DEFAULT-DAG: exp
; DEFAULT: exp
; DEFAULT-NEXT: s_endpgm
define amdgpu_vs void @main(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg1, ptr addrspace(4) inreg %arg2, ptr addrspace(4) inreg %arg3, ptr addrspace(4) inreg %arg4, i32 inreg %arg5, i32 %arg6, i32 %arg7, i32 %arg8, i32 %arg9, ptr addrspace(4) inreg %constptr) #0 {
main_body:
  %tmp10 = load <16 x i8>, ptr addrspace(4) %arg3, !tbaa !0
  %tmp10.cast = bitcast <16 x i8> %tmp10 to <4 x i32>
  %tmp11 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %tmp10.cast, i32 %arg6, i32 0, i32 0, i32 0)
  %tmp12 = extractelement <4 x float> %tmp11, i32 0
  %tmp13 = extractelement <4 x float> %tmp11, i32 1
  call void @llvm.amdgcn.s.barrier() #1
  %tmp14 = extractelement <4 x float> %tmp11, i32 2
  %tmp15 = load float, ptr addrspace(4) %constptr, align 4
  %tmp16 = getelementptr <16 x i8>, ptr addrspace(4) %arg3, i32 1
  %tmp17 = load <16 x i8>, ptr addrspace(4) %tmp16, !tbaa !0
  %tmp17.cast = bitcast <16 x i8> %tmp17 to <4 x i32>
  %tmp18 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %tmp17.cast, i32 %arg6, i32 0, i32 0, i32 0)
  %tmp19 = extractelement <4 x float> %tmp18, i32 0
  %tmp20 = extractelement <4 x float> %tmp18, i32 1
  %tmp21 = extractelement <4 x float> %tmp18, i32 2
  %tmp22 = extractelement <4 x float> %tmp18, i32 3
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %tmp19, float %tmp20, float %tmp21, float %tmp22, i1 false, i1 false) #0
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float %tmp12, float %tmp13, float %tmp14, float %tmp15, i1 true, i1 false) #0
  ret void
}

; ILPMAX-LABEL: {{^}}main2:
; ILPMAX: s_load_dwordx8
; ILPMAX: s_waitcnt lgkmcnt(0)
; ILPMAX: buffer_load
; ILPMAX: buffer_load
; ILPMAX: s_waitcnt vmcnt(0)
; ILPMAX: exp pos0
; ILPMAX-NEXT: exp param0
; ILPMAX: s_endpgm
define amdgpu_vs void @main2(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg1, ptr addrspace(4) inreg %arg2, ptr addrspace(4) inreg %arg3, ptr addrspace(4) inreg %arg4, i32 inreg %arg5, i32 inreg %arg6, i32 %arg7, i32 %arg8, i32 %arg9, i32 %arg10) #0 {
main_body:
  %tmp11 = load <16 x i8>, ptr addrspace(4) %arg4, align 16, !tbaa !0
  %tmp12 = add i32 %arg5, %arg7
  %tmp11.cast = bitcast <16 x i8> %tmp11 to <4 x i32>
  %tmp13 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %tmp11.cast, i32 %tmp12, i32 0, i32 0, i32 0)
  %tmp14 = extractelement <4 x float> %tmp13, i32 0
  %tmp15 = extractelement <4 x float> %tmp13, i32 1
  %tmp16 = extractelement <4 x float> %tmp13, i32 2
  %tmp17 = extractelement <4 x float> %tmp13, i32 3
  %tmp18 = getelementptr [16 x <16 x i8>], ptr addrspace(4) %arg4, i64 0, i64 1
  %tmp19 = load <16 x i8>, ptr addrspace(4) %tmp18, align 16, !tbaa !0
  %tmp20 = add i32 %arg5, %arg7
  %tmp19.cast = bitcast <16 x i8> %tmp19 to <4 x i32>
  %tmp21 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %tmp19.cast, i32 %tmp20, i32 0, i32 0, i32 0)
  %tmp22 = extractelement <4 x float> %tmp21, i32 0
  %tmp23 = extractelement <4 x float> %tmp21, i32 1
  %tmp24 = extractelement <4 x float> %tmp21, i32 2
  %tmp25 = extractelement <4 x float> %tmp21, i32 3
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float %tmp14, float %tmp15, float %tmp16, float %tmp17, i1 false, i1 false) #0
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %tmp22, float %tmp23, float %tmp24, float %tmp25, i1 true, i1 false) #0
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32>, i32, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readonly }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", !2}
!2 = !{!"tbaa root"}
