; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; With untyped pointers enabled a function pointer must stay typed, since an
; untyped pointer cannot express the function type. A select between two function
; pointers then has matching typed operands and result.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FNTY:]] = OpTypeFunction %[[#I32]] %[[#I32]] %[[#I32]]
; CHECK-DAG: %[[#FPTY:]] = OpTypePointer CodeSectionINTEL %[[#FNTY]]
; CHECK: %[[#BAR:]] = OpConstantFunctionPointerINTEL %[[#FPTY]]
; CHECK: %[[#BAZ:]] = OpConstantFunctionPointerINTEL %[[#FPTY]]
; CHECK: OpSelect %[[#]] %[[#]] %[[#BAZ]] %[[#BAR]]
; CHECK: OpFunctionPointerCallINTEL
define spir_func i32 @caller(i1 %c, i32 %a, i32 %b) {
  %fp = select i1 %c, ptr @bar, ptr @baz
  %r = call spir_func i32 %fp(i32 %a, i32 %b)
  ret i32 %r
}
define spir_func i32 @bar(i32 %a, i32 %b) {
  %s = add i32 %a, %b
  ret i32 %s
}
define spir_func i32 @baz(i32 %a, i32 %b) {
  %s = mul i32 %a, %b
  ret i32 %s
}
