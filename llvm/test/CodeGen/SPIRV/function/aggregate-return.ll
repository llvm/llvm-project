; Test that functions returning aggregate types (struct/array) are lowered
; correctly to SPIR-V without crashing the IR verifier.
; See https://github.com/llvm/llvm-project/issues/208899
;
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#FLOAT]] %[[#]]
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#FLOAT]] %[[#FLOAT]]

; A function returning an array type.
; CHECK: OpFunction
; CHECK: OpReturnValue
define [1 x float] @array_return() {
  %arr = insertvalue [1 x float] undef, float 1.0, 0
  ret [1 x float] %arr
}

; A function returning a struct type.
; CHECK: OpFunction
; CHECK: OpReturnValue
define { float, float } @struct_return() {
  %s = insertvalue { float, float } undef, float 0.0, 0
  %s2 = insertvalue { float, float } %s, float 1.0, 1
  ret { float, float } %s2
}

; A function returning an array obtained via extractvalue from a call.
define [1 x float] @extractvalue_return() {
  %call = call [1 x float] @array_return()
  %arr = extractvalue [1 x float] %call, 0
  %result = insertvalue [1 x float] undef, float %arr, 0
  ret [1 x float] %result
}
