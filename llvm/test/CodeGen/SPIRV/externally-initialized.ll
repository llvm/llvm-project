; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s --check-prefix=CHECK-AMDGCNSPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpName %[[#G:]] "G"
; CHECK-SPIRV-NOT: OpDecorate %[[#G]] ReferencedIndirectlyINTEL
; CHECK-SPIRV-DAG: %[[#G]] = OpVariable

; CHECK-AMDGCNSPIRV: OpExtension "SPV_INTEL_global_variable_host_access"
; CHECK-AMDGCNSPIRV: OpName %[[#G:]] "G"
; CHECK-AMDGCNSPIRV: OpDecorate %[[#G]] HostAccessINTEL 3 "G"
; CHECK-AMDGCNSPIRV-DAG: %[[#G]] = OpVariable


@G = external addrspace(1) externally_initialized global i32

define spir_func i32 @foo() {
  %r = load i32, ptr addrspace(1) @G
  ret i32 %r
}
