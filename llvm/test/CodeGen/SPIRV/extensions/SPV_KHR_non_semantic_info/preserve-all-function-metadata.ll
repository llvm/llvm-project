; Adapted from the SPIRV-LLVM-Translator --spirv-preserve-auxdata forward-path
; tests. Numeric metadata operands become OpConstants; string metadata operands
; become OpStrings.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

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
; CHECK-DAG: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn0]] %[[#MDNameVal]] %[[#Const]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn1]] %[[#MDName]] %[[#MDValue]]

target triple = "spir64-unknown-unknown"

define spir_func void @test_val() !foo !1 {
  ret void
}

define spir_func void @test_string() !bar !2 {
  ret void
}

!1 = !{i32 5}
!2 = !{!"baz"}
