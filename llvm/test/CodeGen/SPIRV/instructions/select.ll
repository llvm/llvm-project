; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG:  OpName [[SCALARi32:%.+]] "select_i32"
; CHECK-DAG:  OpName [[VEC2i32:%.+]] "select_i32v2"
; CHECK-DAG:  OpName [[VEC2i32v2:%.+]] "select_v2i32v2"

; CHECK:      [[SCALARi32]] = OpFunction
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpSelect {{%.+}} [[C]] [[T]] [[F]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @select_i32(i1 %c, i32 %t, i32 %f) {
  %r = select i1 %c, i32 %t, i32 %f
  ret i32 %r
}

; CHECK:      [[VEC2i32]] = OpFunction
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpSelect {{%.+}} [[C]] [[T]] [[F]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i32> @select_i32v2(i1 %c, <2 x i32> %t, <2 x i32> %f) {
  %r = select i1 %c, <2 x i32> %t, <2 x i32> %f
  ret <2 x i32> %r
}

; CHECK:      [[VEC2i32v2]] = OpFunction
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpSelect {{%.+}} [[C]] [[T]] [[F]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i32> @select_v2i32v2(<2 x i1> %c, <2 x i32> %t, <2 x i32> %f) {
  %r = select <2 x i1> %c, <2 x i32> %t, <2 x i32> %f
  ret <2 x i32> %r
}
