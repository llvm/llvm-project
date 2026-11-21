; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V8F32:]] = OpTypeVector %[[#F32]] 8
; CHECK-DAG: %[[#V16F32:]] = OpTypeVector %[[#F32]] 16

; Power-of-two vectors up to <16 x float> are legal as is.
; CHECK-LABEL: Begin function frexp_v8f32
; CHECK: OpExtInst %[[#V8F32]] %[[#]] frexp
; CHECK-NOT: OpExtInst
; CHECK: OpFunctionEnd
define <8 x float> @frexp_v8f32(<8 x float> %a) {
  %r = call { <8 x float>, <8 x i32> } @llvm.frexp.v8f32.v8i32(<8 x float> %a)
  %m = extractvalue { <8 x float>, <8 x i32> } %r, 0
  ret <8 x float> %m
}

; Vectors with a non-power-of-two element count are scalarized.
; CHECK-LABEL: Begin function frexp_v5f32
; CHECK-COUNT-5: OpExtInst %[[#F32]] %[[#]] frexp
; CHECK-NOT: OpExtInst
; CHECK: OpFunctionEnd
define <5 x float> @frexp_v5f32(<5 x float> %a) {
  %r = call { <5 x float>, <5 x i32> } @llvm.frexp.v5f32.v5i32(<5 x float> %a)
  %m = extractvalue { <5 x float>, <5 x i32> } %r, 0
  ret <5 x float> %m
}

; Vectors larger than <16 x float> are split.
; CHECK-LABEL: Begin function frexp_v17f32
; CHECK: OpExtInst %[[#V16F32]] %[[#]] frexp
; CHECK: OpExtInst %[[#F32]] %[[#]] frexp
; CHECK-NOT: OpExtInst
; CHECK: OpFunctionEnd
define <17 x float> @frexp_v17f32(<17 x float> %a) {
  %r = call { <17 x float>, <17 x i32> } @llvm.frexp.v17f32.v17i32(<17 x float> %a)
  %m = extractvalue { <17 x float>, <17 x i32> } %r, 0
  ret <17 x float> %m
}
