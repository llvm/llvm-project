; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpDecorate %[[#FN:]] LinkageAttributes "" Export
; CHECK-NOT: OpDecorate %[[#]] LinkageAttributes "" Export
; CHECK: %[[#FN]] = OpFunction

define void @0() {
  ret void
}

define spir_kernel void @kernel() {
  ret void
}
