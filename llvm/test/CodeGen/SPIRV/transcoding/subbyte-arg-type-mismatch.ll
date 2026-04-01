; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

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
