; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; This test only intends to check the vstoren builtin name resolution.
; The calls to the OpenCL builtins are not valid and will not pass SPIR-V validation.

; CHECK-DAG: %[[#IMPORT:]] = OpExtInstImport "OpenCL.std"

; CHECK-DAG: %[[#VOID:]] = OpTypeVoid
; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#INT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#VINT8:]] = OpTypeVector %[[#INT8]] 2
; CHECK-DAG: %[[#PTRINT8:]] = OpTypePointer CrossWorkgroup %[[#INT8]]

; CHECK: %[[#DATA:]] = OpFunctionParameter %[[#VINT8]]
; CHECK: %[[#OFFSET:]] = OpFunctionParameter %[[#INT64]]
; CHECK: %[[#ADDRESS:]] = OpFunctionParameter %[[#PTRINT8]]

define spir_kernel void @test_fn(<2 x i8> %data, i64 %offset, ptr addrspace(1) %address) {
; CHECK: %[[#]] = OpExtInst %[[#VOID]] %[[#IMPORT]] vstoren %[[#DATA]] %[[#OFFSET]] %[[#ADDRESS]]
  call spir_func void @_Z7vstore2Dv2_cmPU3AS1c(<2 x i8> %data, i64 %offset, ptr addrspace(1) %address)
  ret void
}

declare spir_func void @_Z7vstore2Dv2_cmPU3AS1c(<2 x i8>, i64, ptr addrspace(1))
