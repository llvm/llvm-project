; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort,+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort,+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; The service function emitting @ext must not leave an OpAbortKHR behind.

; CHECK-NOT: AbortKHR
; CHECK: OpDecorate %[[#EXT:]] LinkageAttributes "ext" Import
; CHECK: %[[#EXT]] = OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpFunctionEnd
; CHECK-NOT: AbortKHR

declare spir_func i32 @ext(i32)

define spir_func void @f(ptr addrspace(1) %out) {
  store ptr @ext, ptr addrspace(1) %out
  ret void
}
