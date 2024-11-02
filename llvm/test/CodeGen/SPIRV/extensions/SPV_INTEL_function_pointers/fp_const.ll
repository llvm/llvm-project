; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Int8
; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpCapability Int64
; CHECK: OpExtension "SPV_INTEL_function_pointers"
; CHECK-DAG: %[[TyInt8:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyInt64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyFunFp:.*]] = OpTypeFunction %[[TyVoid]] %[[TyInt64]]
; CHECK-DAG: %[[ConstInt64:.*]] = OpConstant %[[TyInt64]] 42
; CHECK-DAG: %[[TyPtrFunFp:.*]] = OpTypePointer Function %[[TyFunFp]]
; CHECK-DAG: %[[ConstFunFp:.*]] = OpConstantFunctionPointerINTEL %[[TyPtrFunFp]] %[[DefFunFp:.*]]
; CHECK: %[[FunPtr1:.*]] = OpBitcast %[[#]] %[[ConstFunFp]]
; CHECK: %[[FunPtr2:.*]] = OpLoad %[[#]] %[[FunPtr1]]
; CHECK: OpFunctionPointerCallINTEL %[[TyInt64]] %[[FunPtr2]] %[[ConstInt64]]
; CHECK: OpReturn
; CHECK: OpFunctionEnd
; CHECK: %[[DefFunFp]] = OpFunction %[[TyVoid]] None %[[TyFunFp]]

target triple = "spir64-unknown-unknown"

define spir_kernel void @test() {
entry:
  %0 = load ptr, ptr @foo
  %1 = call i64 %0(i64 42)
  ret void
}

define void @foo(i64 %a) {
entry:
  ret void
}
