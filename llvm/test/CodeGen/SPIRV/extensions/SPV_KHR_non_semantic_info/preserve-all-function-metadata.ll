; Adapted from the SPIRV-LLVM-Translator --spirv-preserve-auxdata forward-path
; tests. The backend only records metadata whose operands are all MDStrings, so
; the numeric !{i32 5} metadata on @test_val is dropped; only @test_string is
; emitted.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; CHECK: %[[#Import:]] = OpExtInstImport "NonSemantic.AuxData"

; CHECK-DAG: %[[#MDName:]] = OpString "bar"
; CHECK-DAG: %[[#MDValue:]] = OpString "baz"
; CHECK-DAG: %[[#Fcn1:]] = OpString "test_string"

; CHECK-DAG: %[[#VoidT:]] = OpTypeVoid

; Numeric function metadata is not representable as OpStrings and is skipped.
; CHECK-NOT: OpString "test_val"

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
