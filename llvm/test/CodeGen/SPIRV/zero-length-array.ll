; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#void:]] = OpTypeVoid
; CHECK: %[[#type:]] = OpTypeInt 32 0
; CHECK: %[[#ext:]] = OpConstant %[[#type]] 0
; CHECK: %[[#]] = OpFunction %[[#void]]

define spir_func void @_Z3foov() {
entry:
  %i = alloca [0 x i32], align 4
  ret void
}
