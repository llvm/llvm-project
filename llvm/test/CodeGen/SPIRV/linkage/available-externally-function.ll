; Without SPV_KHR_non_semantic_info: emitted as plain external definitions
; (original linkage has no native SPIR-V representation).

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=NOAUX
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=NOAUX
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; NOAUX-NOT: OpExtension "SPV_KHR_non_semantic_info"
; NOAUX-NOT: NonSemantic.AuxData
; NOAUX-DAG: OpName %[[#ae_func:]] "ae_func"
; NOAUX-DAG: OpName %[[#caller:]] "caller"
; NOAUX-NOT: OpDecorate %[[#ae_func]] LinkageAttributes
; NOAUX:     %[[#ae_func]] = OpFunction
; NOAUX:     OpFunctionEnd
; NOAUX:     %[[#caller]] = OpFunction
; NOAUX:     OpFunctionCall %[[#]] %[[#ae_func]]

; With SPV_KHR_non_semantic_info: linkage preserved via a
; NonSemantic.AuxData::Linkage annotation for round-trip recovery.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=AUX
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=AUX
;; spirv-val rejects the AuxData forward-ref without
;; SPV_KHR_relaxed_extended_instruction; matches Translator behavior.

; AUX-DAG: OpExtension "SPV_KHR_non_semantic_info"
; AUX-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; AUX-DAG: OpName %[[#ae_func:]] "ae_func"
; AUX-DAG: OpName %[[#caller:]] "caller"
; AUX-NOT: OpDecorate %[[#ae_func]] LinkageAttributes
; AUX-DAG: %[[#void:]] = OpTypeVoid
; AUX-DAG: %[[#i32:]] = OpTypeInt 32 0
; AUX-DAG: %[[#zero:]] = OpConstant %[[#i32]] 0
; AUX:     %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#ae_func]] %[[#zero]]
; AUX:     %[[#ae_func]] = OpFunction
; AUX:     OpFunctionEnd
; AUX:     %[[#caller]] = OpFunction
; AUX:     OpFunctionCall %[[#]] %[[#ae_func]]

define available_externally spir_func i32 @ae_func(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define spir_kernel void @caller(ptr addrspace(1) %out, i32 %n) {
entry:
  %v = call spir_func i32 @ae_func(i32 %n)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
