; RUN: llc -mtriple=spirv64-- --spirv-ext=all < %s | FileCheck %s --check-prefix=CHECK
; No need to validate the output of the first command, we just want to ensure that we are on a path that triggers the use of SPV_KHR_float_controls2

; RUN: llc -mtriple=spirv64-amd-amdhsa --spirv-ext=all < %s | FileCheck %s --check-prefix=CHECK-AMD
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-amd-amdhsa --spirv-ext=all < %s -filetype=obj | spirv-val %}

; RUN: llc -mtriple=spirv64-amd-amdhsa --spirv-ext=+SPV_KHR_float_controls2 < %s | FileCheck %s --check-prefix=CHECK-AMD
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-amd-amdhsa --spirv-ext=+SPV_KHR_float_controls2 < %s -filetype=obj | spirv-val %}

; Check that SPV_KHR_float_controls2 is not present when the target is AMD.
; AMD's SPIRV implementation uses the translator to get bitcode from SPIRV,
; which at the moment doesn't implement the SPV_KHR_float_controls2 extension.

; CHECK: SPV_KHR_float_controls2
; CHECK-AMD-NOT: SPV_KHR_float_controls2

define spir_kernel void @foo(float %a, float %b, ptr addrspace(1) %out) {
entry:
  ; Use contract to trigger a use of SPV_KHR_float_controls2
  %r1 = fadd contract float %a, %b
  store volatile float %r1, ptr addrspace(1) %out
  ret void
}
