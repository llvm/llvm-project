; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpCapability Int64
; CHECK: OpExtension "SPV_INTEL_function_pointers"

; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyInt64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyFun:.*]] = OpTypeFunction %[[TyInt64]] %[[TyInt64]]
; CHECK-DAG: %[[TyPtrFunCodeSection:.*]] = OpTypePointer CodeSectionINTEL %[[TyFun]]
; CHECK-DAG: %[[TyPtrFun:.*]] = OpTypePointer Function %[[TyFun]]
; CHECK-DAG: %[[TyPtrPtrFun:.*]] = OpTypePointer Function %[[TyPtrFun]]
; CHECK-DAG: %[[TyInt8:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[TyPtrInt8:.*]] = OpTypePointer Function %[[TyInt8]]
; CHECK-DAG: %[[TyPtrPtrInt8:.*]] = OpTypePointer Function %[[TyPtrInt8]]
; CHECK-DAG: %[[TyPtrPtrFunCodeSection:.*]] = OpTypePointer Function %[[TyPtrFunCodeSection]]
; CHECK-DAG: %[[ConstFunFp:.*]] = OpConstantFunctionPointerINTEL %[[TyPtrFunCodeSection]] %[[DefFunFp:.*]]
; CHECK: OpFunction
; CHECK: %[[Var:.*]] = OpVariable %[[TyPtrPtrInt8]] Function
; CHECK: %[[Cast1:.*]] = OpBitcast %[[TyPtrPtrFun]] %[[Var]]
; CHECK: %[[Cast2:.*]] = OpBitcast %[[TyPtrPtrFunCodeSection]] %[[Cast1]]
; CHECK: OpStore %[[Cast2]] %[[ConstFunFp]] Aligned 4
; CHECK: %[[Cast3:.*]] = OpBitcast %[[TyPtrPtrFun]] %[[Var]]
; CHECK: %[[FP:.*]] = OpLoad %[[TyPtrFun]] %[[Cast3]] Aligned 4
; CHECK: OpFunctionPointerCallINTEL %[[TyInt64]] %[[FP]] %[[#]]
; CHECK: OpFunctionEnd

; CHECK: %[[DefFunFp]] = OpFunction %[[TyInt64]] None %[[TyFun]]

define spir_kernel void @test() {
entry:
  %fp = alloca ptr
  store ptr @foo, ptr %fp
  %tocall = load ptr, ptr %fp
  %res = call i64 %tocall(i64 42)
  ret void
}

define i64 @foo(i64 %a) {
entry:
  ret i64 %a
}
