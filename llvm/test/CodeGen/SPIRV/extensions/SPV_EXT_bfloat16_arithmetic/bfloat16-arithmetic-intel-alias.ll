; SPV_INTEL_bfloat16_arithmetic is the legacy name of SPV_EXT_bfloat16_arithmetic and enables
; the same capability. It is only used when SPV_EXT_bfloat16_arithmetic is not enabled.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s --check-prefix=CHECK,CHECK-INTEL
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s --check-prefix=CHECK,CHECK-EXT
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_bfloat16_arithmetic,+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s --check-prefix=CHECK,CHECK-EXT
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=all %s -o - | FileCheck %s --check-prefix=CHECK,CHECK-EXT

; SPIRV-Tools accept the BFloat16Arithmetic capability with SPV_EXT_bfloat16_arithmetic only.
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_bfloat16_arithmetic,+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=all %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability BFloat16TypeKHR
; CHECK-DAG: OpCapability BFloat16ArithmeticEXT
; CHECK-DAG: OpExtension "SPV_KHR_bfloat16"

; CHECK-INTEL-DAG: OpExtension "SPV_INTEL_bfloat16_arithmetic"
; CHECK-INTEL-NOT: OpExtension "SPV_EXT_bfloat16_arithmetic"

; CHECK-EXT-DAG: OpExtension "SPV_EXT_bfloat16_arithmetic"
; CHECK-EXT-NOT: OpExtension "SPV_INTEL_bfloat16_arithmetic"

define spir_func bfloat @add(bfloat %x, bfloat %y) {
entry:
; CHECK: OpFAdd
  %r = fadd bfloat %x, %y
  ret bfloat %r
}
