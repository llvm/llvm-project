; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; An aggregate extractvalue feeding an indirect call used to crash the verifier.

%struct.Inner = type { float, float }
%struct.Outer = type { %struct.Inner, %struct.Inner }

; CHECK: %[[#Inner:]] = OpTypeStruct
; CHECK: %[[#Outer:]] = OpTypeStruct %[[#Inner]] %[[#Inner]]
; CHECK: %[[#Null:]] = OpConstantNull %[[#Outer]]

; CHECK: OpFunction
; CHECK: %[[#Field:]] = OpCompositeExtract %[[#Inner]] %[[#Null]] 1
; CHECK: %[[#]] = OpFunctionPointerCallINTEL %[[#]] %[[#]] %[[#Field]]
define spir_func i32 @f(ptr %fp) {
entry:
  %a = extractvalue %struct.Outer zeroinitializer, 1
  %r = call spir_func i32 %fp(%struct.Inner %a)
  ret i32 %r
}
