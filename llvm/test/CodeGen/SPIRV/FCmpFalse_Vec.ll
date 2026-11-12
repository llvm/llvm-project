; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#BoolTy:]] = OpTypeBool
; CHECK: %[[#VecTy:]] = OpTypeVector %[[#BoolTy]] 4
; CHECK: %[[#NullVec:]] = OpConstantNull %[[#VecTy]]
; CHECK: OpReturnValue %[[#NullVec]]

define spir_func <4 x i1> @test(<4 x float> %a) {
 %compare = fcmp false <4 x float> %a, %a
 ret <4 x i1> %compare
}
