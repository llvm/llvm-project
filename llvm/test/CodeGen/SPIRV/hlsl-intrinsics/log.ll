; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @log_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Log %[[#]]
  %elt.log = call float @llvm.log.f32(float %a)
  ret float %elt.log
}

define noundef half @log_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Log %[[#]]
  %elt.log = call half @llvm.log.f16(half %a)
  ret half %elt.log
}

declare half @llvm.log.f16(half)
declare float @llvm.log.f32(float)
