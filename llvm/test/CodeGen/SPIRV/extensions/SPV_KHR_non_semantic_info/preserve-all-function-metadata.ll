; Function metadata is emitted as NonSemantic.AuxData records. Numeric operands
; become OpConstants; string operands become OpStrings.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Import:]] = OpExtInstImport "NonSemantic.AuxData"

; CHECK-DAG: %[[#MDNameVal:]] = OpString "foo"
; CHECK-DAG: %[[#MDName:]] = OpString "bar"
; CHECK-DAG: %[[#MDValue:]] = OpString "baz"

; The Target operand is the real OpFunction <id>, not an OpString name.
; CHECK-DAG: OpName %[[#Fcn0:]] "test_val"
; CHECK-DAG: OpName %[[#Fcn1:]] "test_string"

; CHECK-DAG: %[[#VoidT:]] = OpTypeVoid
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const:]] = OpConstant %[[#I32]] 5

; Numeric metadata operand is emitted as an OpConstant id.
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn0]] %[[#MDNameVal]] %[[#Const]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn1]] %[[#MDName]] %[[#MDValue]]

target triple = "spir64-unknown-unknown"

define spir_func void @test_val() !foo !1 {
  ret void
}

define spir_func void @test_string() !bar !2 {
  ret void
}

!1 = !{i32 5}
!2 = !{!"baz"}
