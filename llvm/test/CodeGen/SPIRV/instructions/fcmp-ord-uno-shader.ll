; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; OpOrdered and OpUnordered require the Kernel capability, so shaders use
; OpIsNan instead.

; CHECK-DAG: OpName [[ORD:%.*]] "test_ord"
; CHECK-DAG: OpName [[UNO:%.*]] "test_uno"
; CHECK-DAG: OpName [[v4ORD:%.*]] "test_v4_ord"
; CHECK-DAG: OpName [[v4UNO:%.*]] "test_v4_uno"

; CHECK:      [[ORD]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[NANA:%.*]] = OpIsNan [[BOOL:%.+]] [[A]]
; CHECK-NEXT: [[NANB:%.*]] = OpIsNan [[BOOL]] [[B]]
; CHECK-NEXT: [[UNOR:%.*]] = OpLogicalOr [[BOOL]] [[NANA]] [[NANB]]
; CHECK-NEXT: [[R:%.*]] = OpLogicalNot [[BOOL]] [[UNOR]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ord(float %a, float %b) {
  %r = fcmp ord float %a, %b
  ret i1 %r
}

; CHECK:      [[UNO]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[NANA:%.*]] = OpIsNan [[BOOL]] [[A]]
; CHECK-NEXT: [[NANB:%.*]] = OpIsNan [[BOOL]] [[B]]
; CHECK-NEXT: [[R:%.*]] = OpLogicalOr [[BOOL]] [[NANA]] [[NANB]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_uno(float %a, float %b) {
  %r = fcmp uno float %a, %b
  ret i1 %r
}

; CHECK:      [[v4ORD]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[NANA:%.*]] = OpIsNan [[V4BOOL:%.+]] [[A]]
; CHECK-NEXT: [[NANB:%.*]] = OpIsNan [[V4BOOL]] [[B]]
; CHECK-NEXT: [[UNOR:%.*]] = OpLogicalOr [[V4BOOL]] [[NANA]] [[NANB]]
; CHECK-NEXT: [[R:%.*]] = OpLogicalNot [[V4BOOL]] [[UNOR]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i1> @test_v4_ord(<4 x float> %a, <4 x float> %b) {
  %r = fcmp ord <4 x float> %a, %b
  ret <4 x i1> %r
}

; CHECK:      [[v4UNO]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[NANA:%.*]] = OpIsNan [[V4BOOL]] [[A]]
; CHECK-NEXT: [[NANB:%.*]] = OpIsNan [[V4BOOL]] [[B]]
; CHECK-NEXT: [[R:%.*]] = OpLogicalOr [[V4BOOL]] [[NANA]] [[NANB]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i1> @test_v4_uno(<4 x float> %a, <4 x float> %b) {
  %r = fcmp uno <4 x float> %a, %b
  ret <4 x i1> %r
}
