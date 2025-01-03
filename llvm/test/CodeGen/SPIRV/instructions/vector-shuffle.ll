; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; TODO: This test currently fails with LLVM_ENABLE_EXPENSIVE_CHECKS enabled
; XFAIL: expensive_checks

; CHECK-DAG: OpName [[SHFv4:%.+]] "shuffle_v4"
; CHECK-DAG: OpName [[INSv4:%.+]] "insert_v4"
; CHECK-DAG: OpName [[EXTv4:%.+]] "extract_v4"
; CHECK-DAG: OpName [[INSv4C:%.+]] "insert_v4C"
; CHECK-DAG: OpName [[EXTv4C:%.+]] "extract_v4C"


; CHECK:      [[SHFv4]] = OpFunction
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpVectorShuffle {{%.+}} [[A]] [[B]] 0 4 3 6
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x float> @shuffle_v4(<8 x float> %A, <8 x float> %B) {
  %r = shufflevector <8 x float> %A, <8 x float> %B, <4 x i32> <i32 0, i32 4, i32 3, i32 6>
  ret <4 x float> %r
}

; CHECK:      [[INSv4]] = OpFunction
; CHECK-NEXT: [[V:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[E:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpVectorInsertDynamic {{%.+}} [[V]] [[E]] [[C]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x float> @insert_v4(<4 x float> %V, float %E, i32 %C) {
  %r = insertelement <4 x float> %V, float %E, i32 %C
  ret <4 x float> %r
}

; CHECK:      [[EXTv4]] = OpFunction
; CHECK-NEXT: [[V:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpVectorExtractDynamic {{%.+}} [[V]] [[C]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define float @extract_v4(<4 x float> %V, i32 %C) {
  %r = extractelement <4 x float> %V, i32 %C
  ret float %r
}

; CHECK:      [[INSv4C]] = OpFunction
; CHECK-NEXT: [[V:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[E:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpCompositeInsert {{%.+}} [[E]] [[V]] 3
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x float> @insert_v4C(<4 x float> %V, float %E) {
  %r = insertelement <4 x float> %V, float %E, i32 3
  ret <4 x float> %r
}

; CHECK:      [[EXTv4C]] = OpFunction
; CHECK-NEXT: [[V:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpCompositeExtract {{%.+}} [[V]] 2
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define float @extract_v4C(<4 x float> %V) {
  %r = extractelement <4 x float> %V, i32 2
  ret float %r
}
