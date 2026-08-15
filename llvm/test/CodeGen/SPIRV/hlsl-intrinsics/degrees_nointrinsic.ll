; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32

; CHECK-LABEL: Begin function fmul_to_degrees
; CHECK: %[[#arg:]] = OpFunctionParameter %[[#float_32]]
; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Degrees %[[#arg]]

define noundef float @fmul_to_degrees(float noundef %x) {
entry:
  %mul = fmul reassoc nnan ninf nsz arcp afn float %x, f0x42652EE1
  ret float %mul
}