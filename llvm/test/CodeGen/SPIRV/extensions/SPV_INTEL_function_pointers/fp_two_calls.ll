; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Int8
; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpCapability Int64
; CHECK: OpExtension "SPV_INTEL_function_pointers"

; CHECK-DAG: OpName %[[fp:.*]] "fp"
; CHECK-DAG: OpName %[[data:.*]] "data"
; CHECK-DAG: OpName %[[bar:.*]] "bar"
; CHECK-DAG: OpName %[[test:.*]] "test"
; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyFloat32:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[TyInt64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyInt8:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[TyPtrInt8:.*]] = OpTypePointer Function %[[TyInt8]]
; CHECK-DAG: %[[TyUncompleteFp:.*]] = OpTypeFunction %[[TyFloat32]] %[[TyPtrInt8]]
; CHECK-DAG: %[[TyPtrUncompleteFp:.*]] = OpTypePointer Function %[[TyUncompleteFp]]
; CHECK-DAG: %[[TyBar:.*]] = OpTypeFunction %[[TyInt64]] %[[TyPtrUncompleteFp]] %[[TyPtrInt8]]
; CHECK-DAG: %[[TyPtrBar:.*]] = OpTypePointer Function %[[TyBar]]
; CHECK-DAG: %[[TyFp:.*]] = OpTypeFunction %[[TyFloat32]] %[[TyPtrBar]]
; CHECK-DAG: %[[TyPtrFp:.*]] = OpTypePointer Function %[[TyFp]]
; CHECK-DAG: %[[TyTest:.*]] = OpTypeFunction %[[TyVoid]] %[[TyPtrFp]] %[[TyPtrInt8]] %[[TyPtrBar]]
; CHECK: %[[test]] = OpFunction %[[TyVoid]] None %[[TyTest]]
; CHECK: %[[fp]] = OpFunctionParameter %[[TyPtrFp]]
; CHECK: %[[data]] = OpFunctionParameter %[[TyPtrInt8]]
; CHECK: %[[bar]] = OpFunctionParameter %[[TyPtrBar]]
; CHECK: OpFunctionPointerCallINTEL %[[TyFloat32]] %[[fp]] %[[bar]]
; CHECK: OpFunctionPointerCallINTEL %[[TyInt64]] %[[bar]] %[[fp]] %[[data]]
; CHECK: OpReturn
; CHECK: OpFunctionEnd

define spir_kernel void @test(ptr %fp, ptr %data, ptr %bar) {
entry:
  %0 = call spir_func float %fp(ptr %bar)
  %1 = call spir_func i64 %bar(ptr %fp, ptr %data)
  ret void
}
