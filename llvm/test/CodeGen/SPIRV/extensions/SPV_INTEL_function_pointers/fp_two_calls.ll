; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Int8
; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpCapability Int64
; CHECK: OpExtension "SPV_INTEL_function_pointers"
; CHECK-DAG: %[[TyInt8:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyFloat32:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[TyInt64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyPtrInt8:.*]] = OpTypePointer Function %[[TyInt8]]
; CHECK-DAG: %[[TyFunFp:.*]] = OpTypeFunction %[[TyFloat32]] %[[TyPtrInt8]]
; CHECK-DAG: %[[TyFunBar:.*]] = OpTypeFunction %[[TyInt64]] %[[TyPtrInt8]] %[[TyPtrInt8]]
; CHECK-DAG: %[[TyPtrFunFp:.*]] = OpTypePointer Function %[[TyFunFp]]
; CHECK-DAG: %[[TyPtrFunBar:.*]] = OpTypePointer Function %[[TyFunBar]]
; CHECK-DAG: %[[TyFunTest:.*]] = OpTypeFunction %[[TyVoid]] %[[TyPtrInt8]] %[[TyPtrInt8]] %[[TyPtrInt8]]
; CHECK: %[[FunTest:.*]] = OpFunction %[[TyVoid]] None %[[TyFunTest]]
; CHECK: %[[ArgFp:.*]] = OpFunctionParameter %[[TyPtrInt8]]
; CHECK: %[[ArgData:.*]] = OpFunctionParameter %[[TyPtrInt8]]
; CHECK: %[[ArgBar:.*]] = OpFunctionParameter %[[TyPtrInt8]]
; CHECK: OpFunctionPointerCallINTEL %[[TyFloat32]] %[[ArgFp]] %[[ArgBar]]
; CHECK: OpFunctionPointerCallINTEL %[[TyInt64]] %[[ArgBar]] %[[ArgFp]] %[[ArgData]]
; CHECK: OpReturn
; CHECK: OpFunctionEnd

target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr %fp, ptr %data, ptr %bar) {
entry:
  %0 = call spir_func float %fp(ptr %bar)
  %1 = call spir_func i64 %bar(ptr %fp, ptr %data)
  ret void
}
