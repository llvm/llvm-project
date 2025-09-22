; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Regression test for issue https://github.com/llvm/llvm-project/issues/152863
; Ensure OpAtomicCompareExchange returns the correct i64 type when used in phi nodes.
; Previously, this would generate invalid SPIR-V where the atomic operation returned
; uint (32-bit) but the phi node expected ulong (64-bit), causing validation errors.

; CHECK-SPIRV-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#Ptr:]] = OpTypePointer CrossWorkgroup %[[#Long]]
; CHECK-SPIRV-DAG: %[[#Zero64:]] = OpConstantNull %[[#Long]]

; Verify that both the phi node and atomic operation use the same i64 type
; CHECK-SPIRV: %[[#ValuePhi:]] = OpPhi %[[#Long]] %[[#Zero64]] %[[#]] %[[#AtomicResult:]] %[[#]]
; CHECK-SPIRV: %[[#AtomicResult]] = OpAtomicCompareExchange %[[#Long]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#Zero64]] %[[#ValuePhi]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-unknown-unknown"

declare i64 @_Z14atomic_cmpxchgPU8CLglobalVlll(ptr addrspace(1), i64, i64)

define spir_kernel void @test_atomic_cmpxchg_phi(ptr addrspace(1) %ptr) {
conversion:
  br label %L6

L6:                                               ; preds = %L6, %conversion
  %value_phi = phi i64 [ 0, %conversion ], [ %1, %L6 ]
  %1 = call i64 @_Z14atomic_cmpxchgPU8CLglobalVlll(ptr addrspace(1) %ptr, i64 %value_phi, i64 0)
  br label %L6
}
