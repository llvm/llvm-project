; This test case checks that LLVM -> SPIR-V translation produces valid
; SPIR-V module, where a global variable, defined with non-default
; address space, have correct non-function storage class.
;
; No additional checks are needed in addition to simple translation
; to SPIR-V. In case of an error newly produced SPIR-V module validation
; would fail due to spirv-val that detects problematic SPIR-V code from
; translator and reports it as the following error:
;
; "Variables can not have a function[7] storage class outside of a function".
;
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Ptr:]] = OpTypePointer CrossWorkgroup %[[#]]
; CHECK: %[[#]] = OpVariable %[[#Ptr]] CrossWorkgroup %[[#]]

@G = addrspace(1) global i1 true

define spir_func i1 @f(i1 %0) {
 store i1 %0, ptr addrspace(1) @G, align 1
 ret i1 %0
}
