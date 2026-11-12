; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#FalseVal:]] = OpConstantFalse %[[#]]
; CHECK: OpReturnValue %[[#FalseVal:]]

define spir_func i1 @f(float %0) {
 %2 = fcmp false float %0, %0
 ret i1 %2
}
