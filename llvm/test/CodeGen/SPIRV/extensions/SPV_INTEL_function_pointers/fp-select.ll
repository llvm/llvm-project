; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; A function pointer lives in the CodeSectionINTEL storage class, so a select
; between two of them has operands and a result of that same pointer type.

; CHECK-DAG: OpName %[[#BAR:]] "bar"
; CHECK-DAG: OpName %[[#BAZ:]] "baz"
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FNTY:]] = OpTypeFunction %[[#I32]] %[[#I32]] %[[#I32]]
; CHECK-DAG: %[[#FPTY:]] = OpTypePointer CodeSectionINTEL %[[#FNTY]]
; CHECK-DAG: %[[#BARFP:]] = OpConstantFunctionPointerINTEL %[[#FPTY]] %[[#BAR]]
; CHECK-DAG: %[[#BAZFP:]] = OpConstantFunctionPointerINTEL %[[#FPTY]] %[[#BAZ]]
; CHECK: %[[#FP:]] = OpSelect %[[#FPTY]] %[[#]] %[[#BARFP]] %[[#BAZFP]]
; CHECK: OpFunctionPointerCallINTEL %[[#I32]] %[[#FP]]

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
