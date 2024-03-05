; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[Arg1:.*]] "arg1"
; CHECK: OpName %[[Arg2:.*]] "arg2"
; CHECK: OpName %[[NotAStaticPoison:.*]] "poison1"
; CHECK: OpName %[[NotAStaticPoison]] "nil0"
; CHECK: OpName %[[StaticPoisonIntFreeze:.*]] "nil1"
; CHECK: OpName %[[StaticPoisonFloatFreeze:.*]] "nil2"
; CHECK: OpName %[[Arg1]] "val1"
; CHECK: OpName %[[Const100:.*]] "val2"
; CHECK: OpName %[[Const100]] "val3"
; CHECK: OpDecorate
; CHECK-DAG: %[[FloatTy:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[ShortTy:.*]] = OpTypeInt 16 0
; CHECK-DAG: %[[IntTy:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[Undef:.*]] = OpUndef %[[ShortTy]]
; CHECK-DAG: %[[Const100]] = OpConstant %[[IntTy]] 100
; CHECK-DAG: %[[StaticPoisonIntFreeze]] = OpConstantNull %[[IntTy]]
; CHECK-DAG: %[[StaticPoisonFloatFreeze]] = OpConstantNull %[[FloatTy]]
; CHECK: %[[Arg1]] = OpFunctionParameter %[[FloatTy]]
; CHECK: %[[NotAStaticPoison]] = OpIAdd %[[ShortTy]] %[[Arg2]] %[[Undef]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @foo(float %arg1, i16 %arg2) {
entry:
  %poison1 = add i16 %arg2, undef
  %nil0 = freeze i16 %poison1
  %nil1 = freeze i32 undef
  %nil2 = freeze float poison
  %val1 = freeze float %arg1
  %val2 = freeze i32 100
  %val3 = freeze i32 %val2
  ret void
}
