; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; A global initializer that is a constant expression over another global must
; stay an OpSpecConstantOp when the referenced global is an OpUntypedVariableKHR.

; CHECK-DAG: OpCapability UntypedPointersKHR
; CHECK-DAG: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PTR_CW:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#PTR_GEN:]] = OpTypeUntypedPointerKHR Generic

; CHECK-DAG: %[[#SRC:]] = OpUntypedVariableKHR %[[#PTR_CW]] CrossWorkgroup %[[#I32]] %[[#]]
; CHECK-DAG: %[[#CAST:]] = OpSpecConstantOp %[[#PTR_GEN]] PtrCastToGeneric %[[#SRC]]
; CHECK-DAG: %[[#]] = OpUntypedVariableKHR %[[#PTR_CW]] CrossWorkgroup %[[#PTR_GEN]] %[[#CAST]]
; CHECK-DAG: %[[#TOINT:]] = OpSpecConstantOp %[[#I64]] ConvertPtrToU %[[#SRC]]
; CHECK-DAG: %[[#]] = OpUntypedVariableKHR %[[#PTR_CW]] CrossWorkgroup %[[#I64]] %[[#TOINT]]

@src = addrspace(1) global i32 0
@gen = addrspace(1) global ptr addrspace(4) addrspacecast (ptr addrspace(1) @src to ptr addrspace(4))
@num = addrspace(1) global i64 ptrtoint (ptr addrspace(1) @src to i64)
