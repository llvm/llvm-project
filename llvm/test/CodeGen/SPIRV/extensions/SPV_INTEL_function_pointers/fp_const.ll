; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpCapability Int64
; CHECK: OpExtension "SPV_INTEL_function_pointers"

; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyInt64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyFun:.*]] = OpTypeFunction %[[TyInt64]] %[[TyInt64]]
; CHECK-DAG: %[[TyPtrFunCodeSection:.*]] = OpTypePointer CodeSectionINTEL %[[TyFun]]
; CHECK-DAG: %[[ConstFunFp:.*]] = OpConstantFunctionPointerINTEL %[[TyPtrFunCodeSection]] %[[DefFunFp:.*]]
; CHECK-DAG: %[[TyPtrFun:.*]] = OpTypePointer Function %[[TyFun]]
; CHECK-DAG: %[[TyPtrPtrFun:.*]] = OpTypePointer Function %[[TyPtrFun]]
; CHECK: OpFunction
; CHECK: %[[Var:.*]] = OpVariable %[[TyPtrPtrFun]] Function
; CHECK: OpStore %[[Var]] %[[ConstFunFp]]
; CHECK: %[[FP:.*]] = OpLoad %[[TyPtrFun]] %[[Var]]
; CHECK: OpFunctionPointerCallINTEL %[[TyInt64]] %[[FP]] %[[#]]
; CHECK: OpFunctionEnd
 
; CHECK: %[[DefFunFp]] = OpFunction %[[TyInt64]] None %[[TyFun]]

target triple = "spir64-unknown-unknown"

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
