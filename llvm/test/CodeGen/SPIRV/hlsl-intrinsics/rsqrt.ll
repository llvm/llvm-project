; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @rsqrt_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] InverseSqrt %[[#]]
  %elt.rsqrt = call float @llvm.spv.rsqrt.f32(float %a)
  ret float %elt.rsqrt
}

define noundef half @rsqrt_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] InverseSqrt %[[#]]
  %elt.rsqrt = call half @llvm.spv.rsqrt.f16(half %a)
  ret half %elt.rsqrt
}

define noundef double @rsqrt_double(double noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] InverseSqrt %[[#]]
  %elt.rsqrt = call double @llvm.spv.rsqrt.f64(double %a)
  ret double %elt.rsqrt
}

declare half @llvm.spv.sqrt.f16(half)
declare float @llvm.spv.sqrt.f32(float)
declare float @llvm.spv.sqrt.f64(float)
