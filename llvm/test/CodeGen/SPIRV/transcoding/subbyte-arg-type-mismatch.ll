; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; This test verifies that sub-byte integer arguments (e.g. i2) don't cause a
; type mismatch during call lowering.
;
; Sub-byte types like i2 are widened to i8 early on, so the vreg gets LLT s8.
;   %vreg = G_ANYEXT s8, ...   ; vreg is s8 after widening

; CHECK: %[[#CharTy:]] = OpTypeInt 8 0
; CHECK: %[[#FnTy:]] = OpTypeFunction %[[#CharTy]] %[[#CharTy]] %[[#CharTy]]
; CHECK: %[[#Fn:]] = OpFunction %[[#CharTy]] None %[[#FnTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#CharTy]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#CharTy]]
; CHECK: %[[#Res:]] = OpIAdd %[[#CharTy]] %[[#A]] %[[#B]]
; CHECK: OpReturnValue %[[#Res]]

define spir_func i2 @add_i2(i2 %a, i2 %b) {
entry:
  %r = add i2 %a, %b
  ret i2 %r
}
