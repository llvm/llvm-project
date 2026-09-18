; available_externally globals are emitted as definitions with Export linkage.
; The original linkage is only preserved as a NonSemantic.AuxData annotation
; when -spirv-preserve-auxdata is passed.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=NOAUX

; NOAUX-NOT: NonSemantic.AuxData
; NOAUX-DAG: OpName %[[#ae_gv:]] "ae_gv"
; NOAUX-DAG: OpDecorate %[[#ae_gv]] LinkageAttributes "ae_gv" Export
; NOAUX:     %[[#ae_gv]] = OpVariable

; With -spirv-preserve-auxdata: Export linkage plus a NonSemantic.AuxData
; Linkage annotation recording the original available_externally linkage.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info --spirv-preserve-auxdata %s -o - | FileCheck %s --check-prefix=AUX

; AUX-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; AUX-DAG: OpName %[[#ae_gv:]] "ae_gv"
; AUX-DAG: OpDecorate %[[#ae_gv]] LinkageAttributes "ae_gv" Export
; AUX-DAG: %[[#void:]] = OpTypeVoid
; AUX-DAG: %[[#i32:]] = OpTypeInt 32 0
; AUX-DAG: %[[#zero:]] = OpConstant %[[#i32]] 0
; AUX-DAG: %[[#ae_gv]] = OpVariable
; AUX:     %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#ae_gv]] %[[#zero]]

@ae_gv = available_externally addrspace(1) global i32 42, align 4

define spir_kernel void @caller(ptr addrspace(1) %out) {
entry:
  %v = load i32, ptr addrspace(1) @ae_gv, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
