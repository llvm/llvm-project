; available_externally functions are emitted as definitions with Export
; linkage. The original linkage is only preserved as a NonSemantic.AuxData
; annotation when -spirv-preserve-auxdata is passed.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=NOAUX
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=NOAUX
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=NOAUX
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Passing the flag without the extension is a fatal error.

; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown --spirv-preserve-auxdata %s -o - 2>&1 | FileCheck %s --check-prefix=ERR
; ERR: -spirv-preserve-auxdata requires the SPV_KHR_non_semantic_info extension to be enabled.

; NOAUX-NOT: OpExtension "SPV_KHR_non_semantic_info"
; NOAUX-NOT: NonSemantic.AuxData
; NOAUX-DAG: OpName %[[#ae_func:]] "ae_func"
; NOAUX-DAG: OpName %[[#caller:]] "caller"
; NOAUX-DAG: OpDecorate %[[#ae_func]] LinkageAttributes "ae_func" Export
; NOAUX:     %[[#ae_func]] = OpFunction
; NOAUX:     OpFunctionEnd
; NOAUX:     %[[#caller]] = OpFunction
; NOAUX:     OpFunctionCall %[[#]] %[[#ae_func]]

; With -spirv-preserve-auxdata: Export linkage plus a NonSemantic.AuxData
; Linkage annotation recording the original available_externally linkage.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info --spirv-preserve-auxdata %s -o - | FileCheck %s --check-prefix=AUX
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info --spirv-preserve-auxdata %s -o - | FileCheck %s --check-prefix=AUX
;; spirv-val rejects the AuxData forward-ref without
;; SPV_KHR_relaxed_extended_instruction; matches Translator behavior.

; AUX-DAG: OpExtension "SPV_KHR_non_semantic_info"
; AUX-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; AUX-DAG: OpName %[[#ae_func:]] "ae_func"
; AUX-DAG: OpName %[[#caller:]] "caller"
; AUX-DAG: OpDecorate %[[#ae_func]] LinkageAttributes "ae_func" Export
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
