; This test is to check that correct virtual register type is created after ptrtoint.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[GlobalValue:.*]] "dev_global"
; CHECK-DAG: %[[TyI64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyStruct:.*]] = OpTypeStruct %[[TyI64]] %[[TyI64]]
; CHECK-DAG: %[[Const128:.*]] = OpConstant %[[TyI64]] 128
; CHECK-DAG: %[[GlobalValue]] = OpVariable
; CHECK-DAG: %[[PtrToInt:.*]] = OpSpecConstantOp %[[TyI64]] ConvertPtrToU %[[GlobalValue]]
; TODO: The following bitcast line looks unneeded and we may expect it to be removed in future
; CHECK-DAG: %[[UseGlobalValue:.*]] = OpSpecConstantOp %[[TyI64]] Bitcast %[[PtrToInt]]
; CHECK-DAG: %[[ConstComposite:.*]] = OpConstantComposite %[[TyStruct]] %[[Const128]] %[[UseGlobalValue]]
; CHECK-DAG: %[[TyPtrStruct:.*]] = OpTypePointer CrossWorkgroup %[[TyStruct]]
; CHECK: OpVariable %[[TyPtrStruct]] CrossWorkgroup %[[ConstComposite]]
; CHECK: OpFunction

@dev_global = addrspace(1) global [2 x i32] zeroinitializer
@__AsanDeviceGlobalMetadata = addrspace(1) global { i64, i64 } { i64 128, i64 ptrtoint (ptr addrspace(1) @dev_global to i64) }

define void @foo() {
entry:
  ret void
}
