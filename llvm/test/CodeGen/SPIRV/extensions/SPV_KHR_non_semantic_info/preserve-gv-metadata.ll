; Adapted from the SPIRV-LLVM-Translator --spirv-preserve-auxdata forward-path
; tests. Global-variable metadata with all-MDString operands is emitted as a
; NonSemantic.AuxData record.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; CHECK: %[[#Import:]] = OpExtInstImport "NonSemantic.AuxData"

; CHECK-DAG: %[[#MD0:]] = OpString "hi"
; CHECK-DAG: %[[#MD1:]] = OpString "there"
; CHECK-DAG: %[[#GVName:]] = OpString "a"
; CHECK-DAG: %[[#MDName:]] = OpString "some.gv.md"

; CHECK-DAG: %[[#VoidT:]] = OpTypeVoid

; CHECK-DAG: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#GVName]] %[[#MDName]] %[[#MD0]] %[[#MD1]]

target triple = "spir64-unknown-unknown"

@a = addrspace(1) global i8 0, !some.gv.md !0

define spir_func void @use() {
  ret void
}

!0 = !{!"hi", !"there"}
