; Adapted from the SPIRV-LLVM-Translator --spirv-preserve-auxdata forward-path
; tests. Global-variable attributes are emitted as NonSemantic.AuxData records.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; CHECK: %[[#Import:]] = OpExtInstImport "NonSemantic.AuxData"

; CHECK-DAG: %[[#GVName:]] = OpString "g"
; CHECK-DAG: %[[#Attr0:]] = OpString "flag"
; CHECK-DAG: %[[#Attr1LHS:]] = OpString "my-gv-attr"
; CHECK-DAG: %[[#Attr1RHS:]] = OpString "7"

; CHECK-DAG: %[[#VoidT:]] = OpTypeVoid

; CHECK-DAG: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#GVName]] %[[#Attr0]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#GVName]] %[[#Attr1LHS]] %[[#Attr1RHS]]

target triple = "spir64-unknown-unknown"

@g = addrspace(1) global i32 0 #0

define spir_func void @use() {
  ret void
}

attributes #0 = { "my-gv-attr"="7" "flag" }
