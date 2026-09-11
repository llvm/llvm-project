; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A nested aggregate extractvalue can produce an aggregate that is then used as
; the base of another insertvalue. The intermediate aggregate must be lowered as
; an i32 SPIR-V value-id before it is passed to llvm.spv.insertv.

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#]] 1
; CHECK-DAG: %[[#INNER:]] = OpTypeArray %[[#I64]] %[[#ONE]]
; CHECK-DAG: %[[#OUTER:]] = OpTypeArray %[[#INNER]] %[[#ONE]]

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#OUTER]]
; CHECK: %[[#X:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#E:]] = OpCompositeExtract %[[#INNER]] %[[#A]] 0
; CHECK: %[[#I:]] = OpCompositeInsert %[[#INNER]] %[[#X]] %[[#E]] 0
; CHECK: %[[#R:]] = OpCompositeInsert %[[#OUTER]] %[[#I]] %[[#A]] 0
; CHECK: OpReturnValue %[[#R]]
define spir_func [1 x [1 x i64]] @f([1 x [1 x i64]] %a, i64 %x) {
  %e = extractvalue [1 x [1 x i64]] %a, 0
  %i = insertvalue [1 x i64] %e, i64 %x, 0
  %r = insertvalue [1 x [1 x i64]] %a, [1 x i64] %i, 0
  ret [1 x [1 x i64]] %r
}
