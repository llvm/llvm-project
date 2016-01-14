; RUN: llc -march=amdgcn -mcpu=SI --misched=si < %s | FileCheck %s

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

define void @main([6 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <4 x i32>] addrspace(2)* byval, [34 x <8 x i32>] addrspace(2)* byval, float inreg, i32 inreg, <2 x i32>, 
<2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, i32, float, float) #0 {
main_body:
  %22 = bitcast [34 x <8 x i32>] addrspace(2)* %3 to <32 x i8> addrspace(2)*
  %23 = load <32 x i8>, <32 x i8> addrspace(2)* %22, align 32, !tbaa !0
  %24 = bitcast [17 x <4 x i32>] addrspace(2)* %2 to <16 x i8> addrspace(2)*
  %25 = load <16 x i8>, <16 x i8> addrspace(2)* %24, align 16, !tbaa !0
  %26 = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %5, <2 x i32> %11)
  %27 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %5, <2 x i32> %11)
  %28 = bitcast float %26 to i32
  %29 = bitcast float %27 to i32
  %30 = insertelement <2 x i32> undef, i32 %28, i32 0
  %31 = insertelement <2 x i32> %30, i32 %29, i32 1
  %32 = call <4 x float> @llvm.SI.sample.v2i32(<2 x i32> %31, <32 x i8> %23, <16 x i8> %25, i32 2)
  %33 = extractelement <4 x float> %32, i32 0
  %34 = extractelement <4 x float> %32, i32 1
  %35 = extractelement <4 x float> %32, i32 2
  %36 = extractelement <4 x float> %32, i32 3
  %37 = call i32 @llvm.SI.packf16(float %33, float %34)
  %38 = bitcast i32 %37 to float
  %39 = call i32 @llvm.SI.packf16(float %35, float %36)
  %40 = bitcast i32 %39 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %38, float %40, float %38, float %40)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.sample.v2i32(<2 x i32>, <32 x i8>, <16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.SI.packf16(float, float) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" "enable-no-nans-fp-math"="true" }
attributes #1 = { nounwind readnone }

!0 = !{!"const", null, i32 1}
