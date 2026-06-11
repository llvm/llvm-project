; -spirv-preserve-auxdata: emit NonSemantic.AuxData for attrs/metadata.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; Off by default: no AuxData records at all, even with the extension enabled.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefix=OFF

; OFF-NOT: OpExtInstImport "NonSemantic.AuxData"
; OFF-NOT: OpString "my-attr"
; OFF-NOT: OpString "nounwind"
; OFF-NOT: OpString "some.md"
; OFF-NOT: OpString "spv.was-available-externally"

; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#akind:]] = OpString "my-attr"
; CHECK-DAG: %[[#aval:]]  = OpString "val"
; CHECK-DAG: %[[#nounwind:]] = OpString "nounwind"
; CHECK-DAG: %[[#mdname:]] = OpString "some.md"
; CHECK-DAG: %[[#mdval:]] = OpString "hello"
; CHECK-DAG: %[[#numname:]] = OpString "num.md"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0

; The Target operand of every record is the real OpFunction / OpVariable <id>,
; not an OpString of the name. Names are only emitted via OpName.
; CHECK-DAG: OpName %[[#fn:]] "fn"
; CHECK-DAG: OpName %[[#gv:]] "gv"

; Internal marker must not leak as a string.
; CHECK-NOT: OpString "spv.was-available-externally"

; Records emit in module order; numeric metadata becomes OpConstant operands.
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#fn]] %[[#akind]] %[[#aval]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#fn]] %[[#nounwind]]
; CHECK-DAG: %[[#mdconst:]] = OpConstant %[[#i32]] 5
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#gv]] %[[#mdname]] %[[#mdval]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#gv]] %[[#numname]] %[[#mdconst]]

@gv = addrspace(1) global i32 0, align 4, !some.md !0, !num.md !1

define spir_func i32 @fn(i32 %x) "my-attr"="val" nounwind {
  ret i32 %x
}

; @ae has no payload, so no attribute/metadata records are emitted for it.
; CHECK-NOT: OpString "ae"

define available_externally spir_func i32 @ae(i32 %x) {
  ret i32 %x
}

!0 = !{!"hello"}
!1 = !{i32 5}
