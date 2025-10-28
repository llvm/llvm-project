; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[TyInt64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyInt64Ptr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[TyInt64]]
; CHECK-DAG: %[[TyStruct:.*]] = OpTypeStruct %[[TyInt64Ptr]] %[[TyInt64Ptr]]
; CHECK-DAG: %[[ConstStruct:.*]] = OpConstantComposite %[[TyStruct]] %[[ConstField:.*]] %[[ConstField]]
; CHECK-DAG: %[[TyStructPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[TyStruct]]
; CHECK-DAG: OpVariable %[[TyStructPtr]] {{[a-zA-Z]+}} %[[ConstStruct]]

@a = addrspace(1) constant i64 42
@struct = addrspace(1) global {ptr addrspace(1), ptr addrspace(1)} { ptr addrspace(1) @a, ptr addrspace(1) @a }

define spir_kernel void @foo() {
  ret void
}
