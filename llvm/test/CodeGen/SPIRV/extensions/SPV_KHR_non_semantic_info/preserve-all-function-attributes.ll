; Adapted from the SPIRV-LLVM-Translator --spirv-preserve-auxdata forward-path
; tests. The backend only emits, so the reverse-translation checks are dropped.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; CHECK: %[[#Import:]] = OpExtInstImport "NonSemantic.AuxData"

; CHECK-DAG: %[[#Fcn0:]] = OpString "mul_add"
; CHECK-DAG: %[[#Attr0:]] = OpString "foo"
; CHECK-DAG: %[[#Fcn1:]] = OpString "test"
; CHECK-DAG: %[[#Attr1LHS:]] = OpString "bar"
; CHECK-DAG: %[[#Attr1RHS:]] = OpString "baz"

; CHECK-DAG: %[[#VoidT:]] = OpTypeVoid

; CHECK-DAG: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn0]] %[[#Attr0]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn1]] %[[#Attr1LHS]] %[[#Attr1RHS]]

target triple = "spir64-unknown-unknown"

define spir_func void @mul_add() #0 {
  ret void
}

define spir_func void @test() #1 {
  ret void
}

attributes #0 = { "foo" }
attributes #1 = { "bar"="baz" }
