; Check that an int4 cooperative matrix type that is removed before emission
; does not leave its capabilities or extensions in the module.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_int4,+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_int4,+SPV_KHR_cooperative_matrix %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability Int4CooperativeMatrixINTEL
; CHECK-NOT: OpCapability Int4TypeINTEL
; CHECK-NOT: OpCapability CooperativeMatrixKHR
; CHECK-NOT: OpExtension "SPV_INTEL_int4"
; CHECK-NOT: OpExtension "SPV_KHR_cooperative_matrix"
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK: OpFunction

define spir_kernel void @dead_matrix(i1 %c) {
  %m = select i1 %c, target("spirv.CooperativeMatrixKHR", i4, 3, 12, 12, 2) poison, target("spirv.CooperativeMatrixKHR", i4, 3, 12, 12, 2) poison
  ret void
}
