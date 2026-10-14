; Function attributes are emitted as NonSemantic.AuxData records: an OpString
; for the name, plus one for the value if present.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Import:]] = OpExtInstImport "NonSemantic.AuxData"

; CHECK-DAG: %[[#Attr0:]] = OpString "foo"
; CHECK-DAG: %[[#Attr1LHS:]] = OpString "bar"
; CHECK-DAG: %[[#Attr1RHS:]] = OpString "baz"

; The Target operand is the real OpFunction <id>, not an OpString name.
; CHECK-DAG: OpName %[[#Fcn0:]] "mul_add"
; CHECK-DAG: OpName %[[#Fcn1:]] "test"

; CHECK-DAG: %[[#VoidT:]] = OpTypeVoid

; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn0]] %[[#Attr0]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn1]] %[[#Attr1LHS]] %[[#Attr1RHS]]

target triple = "spir64-unknown-unknown"

define spir_func void @mul_add() #0 {
  ret void
}

define spir_func void @test() #1 {
  ret void
}

attributes #0 = { "foo" }
attributes #1 = { "bar"="baz" }
