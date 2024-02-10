; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; This test only intends to check the vloadn builtin name resolution.
; The calls to the OpenCL builtins are not valid and will not pass SPIR-V validation.

; CHECK-DAG: %[[#IMPORT:]] = OpExtInstImport "OpenCL.std"

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#INT16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#INT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#VINT8:]] = OpTypeVector %[[#INT8]] 2
; CHECK-DAG: %[[#VINT16:]] = OpTypeVector %[[#INT16]] 2
; CHECK-DAG: %[[#VINT32:]] = OpTypeVector %[[#INT32]] 2
; CHECK-DAG: %[[#VINT64:]] = OpTypeVector %[[#INT64]] 2
; CHECK-DAG: %[[#VFLOAT:]] = OpTypeVector %[[#FLOAT]] 2
; CHECK-DAG: %[[#PTRINT8:]] = OpTypePointer CrossWorkgroup %[[#INT8]]

; CHECK: %[[#OFFSET:]] = OpFunctionParameter %[[#INT64]]
; CHECK: %[[#ADDRESS:]] = OpFunctionParameter %[[#PTRINT8]]

define spir_kernel void @test_fn(i64 %offset, ptr addrspace(1) %address) {
; CHECK: %[[#]] = OpExtInst %[[#VINT8]] %[[#IMPORT]] vloadn %[[#OFFSET]] %[[#ADDRESS]] 2
  %call1 = call spir_func <2 x i8> @_Z6vload2mPU3AS1Kc(i64 %offset, ptr addrspace(1) %address)
; CHECK: %[[#]] = OpExtInst %[[#VINT16]] %[[#IMPORT]] vloadn %[[#OFFSET]] %[[#ADDRESS]] 2
  %call2 = call spir_func <2 x i16> @_Z6vload2mPU3AS1Ks(i64 %offset, ptr addrspace(1) %address)
; CHECK: %[[#]] = OpExtInst %[[#VINT32]] %[[#IMPORT]] vloadn %[[#OFFSET]] %[[#ADDRESS]] 2
  %call3 = call spir_func <2 x i32> @_Z6vload2mPU3AS1Ki(i64 %offset, ptr addrspace(1) %address)
; CHECK: %[[#]] = OpExtInst %[[#VINT64]] %[[#IMPORT]] vloadn %[[#OFFSET]] %[[#ADDRESS]] 2
  %call4 = call spir_func <2 x i64> @_Z6vload2mPU3AS1Kl(i64 %offset, ptr addrspace(1) %address)
; CHECK: %[[#]] = OpExtInst %[[#VFLOAT]] %[[#IMPORT]] vloadn %[[#OFFSET]] %[[#ADDRESS]] 2
  %call5 = call spir_func <2 x float> @_Z6vload2mPU3AS1Kf(i64 %offset, ptr addrspace(1) %address)
  ret void
}

declare spir_func <2 x i8> @_Z6vload2mPU3AS1Kc(i64, ptr addrspace(1))
declare spir_func <2 x i16> @_Z6vload2mPU3AS1Ks(i64, ptr addrspace(1))
declare spir_func <2 x i32> @_Z6vload2mPU3AS1Ki(i64, ptr addrspace(1))
declare spir_func <2 x i64> @_Z6vload2mPU3AS1Kl(i64, ptr addrspace(1))
declare spir_func <2 x float> @_Z6vload2mPU3AS1Kf(i64, ptr addrspace(1))
