; This test is to check that two functions have different SPIR-V type
; definitions, even though their LLVM function types are identical.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[Fun32:.*]] "tp_arg_i32"
; CHECK-DAG: OpName %[[Fun64:.*]] "tp_arg_i64"
; CHECK-DAG: %[[TyI32:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyPtr32:.*]] = OpTypePointer Function %[[TyI32]]
; CHECK-DAG: %[[TyFun32:.*]] = OpTypeFunction %[[TyVoid]] %[[TyPtr32]]
; CHECK-DAG: %[[TyI64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyPtr64:.*]] = OpTypePointer Function %[[TyI64]]
; CHECK-DAG: %[[TyFun64:.*]] = OpTypeFunction %[[TyVoid]] %[[TyPtr64]]
; CHECK-DAG: %[[Fun32]] = OpFunction %[[TyVoid]] None %[[TyFun32]]
; CHECK-DAG: %[[Fun64]] = OpFunction %[[TyVoid]] None %[[TyFun64]]

define spir_kernel void @tp_arg_i32(ptr %ptr) {
entry:
  store i32 1, ptr %ptr
  ret void
}

define spir_kernel void @tp_arg_i64(ptr %ptr) {
entry:
  store i64 1, ptr %ptr
  ret void
}
