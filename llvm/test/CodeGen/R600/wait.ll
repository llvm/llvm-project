; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -strict-whitespace %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace %s

; CHECK-LABEL: {{^}}main:
; CHECK: s_load_dwordx4
; CHECK: s_load_dwordx4
; CHECK: s_waitcnt lgkmcnt(0){{$}}
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK: s_waitcnt expcnt(0) lgkmcnt(0){{$}}
define void @main(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, <16 x i8> addrspace(2)* inreg %arg3, <16 x i8> addrspace(2)* inreg %arg4, i32 inreg %arg5, i32 %arg6, i32 %arg7, i32 %arg8, i32 %arg9, float addrspace(2)* inreg %constptr) #0 {
main_body:
  %tmp = getelementptr <16 x i8> addrspace(2)* %arg3, i32 0
  %tmp10 = load <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp11 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %tmp10, i32 0, i32 %arg6)
  %tmp12 = extractelement <4 x float> %tmp11, i32 0
  %tmp13 = extractelement <4 x float> %tmp11, i32 1
  call void @llvm.AMDGPU.barrier.global() #1
  %tmp14 = extractelement <4 x float> %tmp11, i32 2
;  %tmp15 = extractelement <4 x float> %tmp11, i32 3
  %tmp15 = load float addrspace(2)* %constptr, align 4 ; Force waiting for expcnt and lgkmcnt
  %tmp16 = getelementptr <16 x i8> addrspace(2)* %arg3, i32 1
  %tmp17 = load <16 x i8> addrspace(2)* %tmp16, !tbaa !0
  %tmp18 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %tmp17, i32 0, i32 %arg6)
  %tmp19 = extractelement <4 x float> %tmp18, i32 0
  %tmp20 = extractelement <4 x float> %tmp18, i32 1
  %tmp21 = extractelement <4 x float> %tmp18, i32 2
  %tmp22 = extractelement <4 x float> %tmp18, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %tmp19, float %tmp20, float %tmp21, float %tmp22)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %tmp12, float %tmp13, float %tmp14, float %tmp15)
  ret void
}

; Function Attrs: noduplicate nounwind
declare void @llvm.AMDGPU.barrier.global() #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.vs.load.input(<16 x i8>, i32, i32) #2

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="1" }
attributes #1 = { noduplicate nounwind }
attributes #2 = { nounwind readnone }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", null}
