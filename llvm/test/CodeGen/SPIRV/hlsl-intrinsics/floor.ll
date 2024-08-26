; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @floor_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Floor %[[#]]
  %elt.floor = call float @llvm.floor.f32(float %a)
  ret float %elt.floor
}

define noundef half @floor_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Floor %[[#]]
  %elt.floor = call half @llvm.floor.f16(half %a)
  ret half %elt.floor
}

declare half @llvm.floor.f16(half)
declare float @llvm.floor.f32(float)
