; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; Check that a void-returning indirect call whose argument is an aggregate
; doesn't crash while the aggregate argument type is temporarily mutated.

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_function_pointers"

; CHECK: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK: %[[#AggTy:]] = OpTypeStruct %[[#Int32Ty]] %[[#Int32Ty]]
; CHECK: %[[#VoidTy:]] = OpTypeVoid
; CHECK: %[[#CalleeTy:]] = OpTypeFunction %[[#VoidTy]] %[[#AggTy]]
; CHECK: %[[#CalleePtrTy:]] = OpTypePointer Generic %[[#CalleeTy]]

; CHECK: %[[#Fp:]] = OpFunctionParameter %[[#CalleePtrTy]]
; CHECK: %[[#Arg:]] = OpFunctionParameter %[[#AggTy]]
; CHECK: OpFunctionPointerCallINTEL %[[#VoidTy]] %[[#Fp]] %[[#Arg]]

%agg = type { i32, i32 }

define spir_func void @caller(ptr addrspace(4) %fp, %agg %a) {
  call addrspace(4) void %fp(%agg %a)
  ret void
}
