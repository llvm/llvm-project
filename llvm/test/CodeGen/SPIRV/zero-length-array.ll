; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#type:]] = OpTypeInt 32 0
; CHECK: %[[#ext:]] = OpTypeRuntimeArray %[[#type]]
; CHECK: %[[#]] = OpTypePointer Function %[[#ext]]

define spir_func void @_Z3foov() {
entry:
  %i = alloca [0 x i32], align 4
  ret void
}
