; -spirv-preserve-auxdata: emit NonSemantic.AuxData for attrs/metadata.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; Off by default: only the Linkage record fires (for the AE function).
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefix=OFF

; OFF-NOT: OpString "my-attr"
; OFF-NOT: OpString "nounwind"
; OFF-NOT: OpString "some.md"
; OFF-NOT: OpString "spv.was-available-externally"

; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#fname:]] = OpString "fn"
; CHECK-DAG: %[[#akind:]] = OpString "my-attr"
; CHECK-DAG: %[[#aval:]]  = OpString "val"
; CHECK-DAG: %[[#nounwind:]] = OpString "nounwind"
; CHECK-DAG: %[[#gname:]] = OpString "gv"
; CHECK-DAG: %[[#mdname:]] = OpString "some.md"
; CHECK-DAG: %[[#mdval:]] = OpString "hello"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0

; Internal marker must not leak as a string.
; CHECK-NOT: OpString "spv.was-available-externally"
; @ae has no payload, so no name OpString.
; CHECK-NOT: OpString "ae"

; Records emit in module order; Linkage's 2nd operand is the shared AE const.
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#fname]] %[[#akind]] %[[#aval]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#fname]] %[[#nounwind]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#gname]] %[[#mdname]] %[[#mdval]]
; CHECK-DAG: %[[#aelink:]] = OpConstant %[[#i32]] 0
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#]] %[[#aelink]]

@gv = global i32 0, align 4, !some.md !0

define spir_func i32 @fn(i32 %x) "my-attr"="val" nounwind {
  ret i32 %x
}

define available_externally spir_func i32 @ae(i32 %x) {
  ret i32 %x
}

!0 = !{!"hello"}
