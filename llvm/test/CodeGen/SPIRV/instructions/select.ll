; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.4-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPV14
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.4-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.4-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPV14
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.4-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.3-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPV13
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.3-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.3 %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.3-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPV13
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.3-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.3 %}

; CHECK-DAG:  OpName [[SCALARi32:%.+]] "select_i32"
; CHECK-DAG:  OpName [[SCALARPTR:%.+]] "select_ptr"
; CHECK-DAG:  OpName [[VEC2i32:%.+]] "select_i32v2"
; CHECK-DAG:  OpName [[VEC2f32:%.+]] "select_f32v2"
; CHECK-DAG:  OpName [[VEC2i32v2:%.+]] "select_v2i32v2"

; CHECK-DAG:  [[BOOL:%.+]] = OpTypeBool
; CHECK-DAG:  [[V2BOOL:%.+]] = OpTypeVector [[BOOL]] 2

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

; CHECK:      [[SCALARPTR]] = OpFunction
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpSelect {{%.+}} [[C]] [[T]] [[F]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define ptr @select_ptr(i1 %c, ptr %t, ptr %f) {
  %r = select i1 %c, ptr %t, ptr %f
  ret ptr %r
}

; CHECK:      [[VEC2i32]] = OpFunction
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; SPV13:      [[CSPLAT:%.+]] = OpCompositeConstruct [[V2BOOL]] [[C]] [[C]]
; SPV13:      [[R:%.+]] = OpSelect {{%.+}} [[CSPLAT]] [[T]] [[F]]
; SPV14-NOT:  OpCompositeConstruct
; SPV14:      [[R:%.+]] = OpSelect {{%.+}} [[C]] [[T]] [[F]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i32> @select_i32v2(i1 %c, <2 x i32> %t, <2 x i32> %f) {
  %r = select i1 %c, <2 x i32> %t, <2 x i32> %f
  ret <2 x i32> %r
}

; CHECK:      [[VEC2f32]] = OpFunction
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; SPV13:      [[CSPLAT:%.+]] = OpCompositeConstruct [[V2BOOL]] [[C]] [[C]]
; SPV13:      [[R:%.+]] = OpSelect {{%.+}} [[CSPLAT]] [[T]] [[F]]
; SPV14-NOT:  OpCompositeConstruct
; SPV14:      [[R:%.+]] = OpSelect {{%.+}} [[C]] [[T]] [[F]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x float> @select_f32v2(i1 %c, <2 x float> %t, <2 x float> %f) {
  %r = select i1 %c, <2 x float> %t, <2 x float> %f
  ret <2 x float> %r
}

; CHECK:      [[VEC2i32v2]] = OpFunction
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK-NOT:  OpCompositeConstruct
; CHECK:      [[R:%.+]] = OpSelect {{%.+}} [[C]] [[T]] [[F]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i32> @select_v2i32v2(<2 x i1> %c, <2 x i32> %t, <2 x i32> %f) {
  %r = select <2 x i1> %c, <2 x i32> %t, <2 x i32> %f
  ret <2 x i32> %r
}
