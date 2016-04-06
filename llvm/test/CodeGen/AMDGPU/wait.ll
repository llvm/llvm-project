; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=DEFAULT
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=DEFAULT
; RUN: llc -march=amdgcn --misched=ilpmax -mcpu=SI -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=ILPMAX
; RUN: llc -march=amdgcn --misched=ilpmax -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace %s --check-prefix=ILPMAX
; The ilpmax scheduler is used for the second test to get the ordering we want for the test.

; DEFAULT-LABEL: {{^}}main:
; DEFAULT: s_load_dwordx4
; DEFAULT: s_load_dwordx4
; DEFAULT: s_waitcnt vmcnt(0)
; DEFAULT: exp
; DEFAULT: s_waitcnt lgkmcnt(0)
; DEFAULT: s_endpgm
define amdgpu_vs void @main(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, <16 x i8> addrspace(2)* inreg %arg3, <16 x i8> addrspace(2)* inreg %arg4, i32 inreg %arg5, i32 %arg6, i32 %arg7, i32 %arg8, i32 %arg9, float addrspace(2)* inreg %constptr) {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg3, i32 0
  %tmp10 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp11 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %tmp10, i32 0, i32 %arg6)
  %tmp12 = extractelement <4 x float> %tmp11, i32 0
  %tmp13 = extractelement <4 x float> %tmp11, i32 1
  call void @llvm.amdgcn.s.barrier() #1
  %tmp14 = extractelement <4 x float> %tmp11, i32 2
;  %tmp15 = extractelement <4 x float> %tmp11, i32 3
  %tmp15 = load float, float addrspace(2)* %constptr, align 4 ; Force waiting for expcnt and lgkmcnt
  %tmp16 = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg3, i32 1
  %tmp17 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp16, !tbaa !0
  %tmp18 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %tmp17, i32 0, i32 %arg6)
  %tmp19 = extractelement <4 x float> %tmp18, i32 0
  %tmp20 = extractelement <4 x float> %tmp18, i32 1
  %tmp21 = extractelement <4 x float> %tmp18, i32 2
  %tmp22 = extractelement <4 x float> %tmp18, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %tmp19, float %tmp20, float %tmp21, float %tmp22)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %tmp12, float %tmp13, float %tmp14, float %tmp15)
  ret void
}

; ILPMAX-LABEL: {{^}}main2:
; ILPMAX: s_load_dwordx4
; ILPMAX: s_waitcnt lgkmcnt(0)
; ILPMAX: buffer_load
; ILPMAX: s_load_dwordx4
; ILPMAX: s_waitcnt lgkmcnt(0)
; ILPMAX: buffer_load
; ILPMAX: s_waitcnt vmcnt(1)
; ILPMAX: s_waitcnt vmcnt(0)
; ILPMAX: s_endpgm

define amdgpu_vs void @main2([6 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <4 x i32>] addrspace(2)* byval, [34 x <8 x i32>] addrspace(2)* byval, [16 x <16 x i8>] addrspace(2)*
byval, i32 inreg, i32 inreg, i32, i32, i32, i32) {
main_body:
  %11 = getelementptr [16 x <16 x i8>], [16 x <16 x i8>] addrspace(2)* %4, i64 0, i64 0
  %12 = load <16 x i8>, <16 x i8> addrspace(2)* %11, align 16, !tbaa !0
  %13 = add i32 %5, %7
  %14 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %12, i32 0, i32 %13)
  %15 = extractelement <4 x float> %14, i32 0
  %16 = extractelement <4 x float> %14, i32 1
  %17 = extractelement <4 x float> %14, i32 2
  %18 = extractelement <4 x float> %14, i32 3
  %19 = getelementptr [16 x <16 x i8>], [16 x <16 x i8>] addrspace(2)* %4, i64 0, i64 1
  %20 = load <16 x i8>, <16 x i8> addrspace(2)* %19, align 16, !tbaa !0
  %21 = add i32 %5, %7
  %22 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %20, i32 0, i32 %21)
  %23 = extractelement <4 x float> %22, i32 0
  %24 = extractelement <4 x float> %22, i32 1
  %25 = extractelement <4 x float> %22, i32 2
  %26 = extractelement <4 x float> %22, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %15, float %16, float %17, float %18)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %23, float %24, float %25, float %26)
  ret void
}


; Function Attrs: convergent nounwind
declare void @llvm.amdgcn.s.barrier() #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.vs.load.input(<16 x i8>, i32, i32) #2

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", null}
