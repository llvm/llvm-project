; Check that a void-returning indirect call whose argument is an aggregate
; doesn't crash while the aggregate argument type is temporarily mutated.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; The -discard-value-names run additionally checks that the per-callsite type
; restoration is keyed independently of value names.
; RUN: llvm-as < %s | llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -discard-value-names -o - | FileCheck %s
; %if spirv-tools %{ llvm-as < %s | llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -discard-value-names -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_function_pointers"

; CHECK: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK: %[[#Agg2Ty:]] = OpTypeStruct %[[#Int32Ty]] %[[#Int32Ty]]
; CHECK: %[[#VoidTy:]] = OpTypeVoid
; CHECK: %[[#VoidCalleeTy:]] = OpTypeFunction %[[#VoidTy]] %[[#Agg2Ty]]
; CHECK: %[[#VoidCalleePtrTy:]] = OpTypePointer Generic %[[#VoidCalleeTy]]
; CHECK: %[[#Callee2Ty:]] = OpTypeFunction %[[#Agg2Ty]] %[[#Agg2Ty]]
; CHECK: %[[#Agg3Ty:]] = OpTypeStruct %[[#Int32Ty]] %[[#Int32Ty]] %[[#Int32Ty]]
; CHECK: %[[#Callee3Ty:]] = OpTypeFunction %[[#Agg3Ty]] %[[#Agg3Ty]]

; CHECK: %[[#Fp:]] = OpFunctionParameter %[[#VoidCalleePtrTy]]
; CHECK: %[[#Arg:]] = OpFunctionParameter %[[#Agg2Ty]]
; CHECK: OpFunctionPointerCallINTEL %[[#VoidTy]] %[[#Fp]] %[[#Arg]]

; CHECK: OpFunctionPointerCallINTEL %[[#Agg2Ty]]
; CHECK: OpFunctionPointerCallINTEL %[[#Agg3Ty]]

%agg2 = type { i32, i32 }
%agg3 = type { i32, i32, i32 }

define spir_func void @caller(ptr addrspace(4) %fp, %agg2 %a) {
  call addrspace(4) void %fp(%agg2 %a)
  ret void
}

define spir_func void @caller_two(ptr addrspace(4) %fp2, ptr addrspace(4) %fp3, %agg2 %a, %agg3 %b) {
  %r2 = call addrspace(4) %agg2 %fp2(%agg2 %a)
  %r3 = call addrspace(4) %agg3 %fp3(%agg3 %b)
  ret void
}
