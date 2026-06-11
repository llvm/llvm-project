; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; XFAIL: *
; Int64Atomics capability is not yet available for Vulkan targets.
; See https://github.com/llvm/llvm-project/issues/202456

; Test lowering of llvm.spv.interlocked.add with i64 to OpAtomicIAdd.

; CHECK-DAG: %[[#ulong:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#scope_wg:]] = OpConstant %[[#uint]] 2
; CHECK-DAG: %[[#scope_dev:]] = OpConstant %[[#uint]] 1
; CHECK-DAG: %[[#mem_wg:]] = OpConstant %[[#uint]] 256
; CHECK-DAG: %[[#mem_uniform:]] = OpConstant %[[#uint]] 64

@gs_i64 = internal addrspace(3) global i64 zeroinitializer
@dev_i64 = external addrspace(11) global i64

; Workgroup (addrspace 3) memory test.

; CHECK-LABEL: Begin function test_i64
define i64 @test_i64(i64 %v) {
entry:
; CHECK: %[[#R:]] = OpAtomicIAdd %[[#ulong]] %[[#]] %[[#scope_wg]] %[[#mem_wg]] %[[#]]
  %r = call i64 @llvm.spv.interlocked.add.i64.p3(ptr addrspace(3) @gs_i64, i64 %v)
  ret i64 %r
}

; Device / StorageBuffer (addrspace 11) memory test.

; CHECK-LABEL: Begin function test_device_i64
define i64 @test_device_i64(i64 %v) {
entry:
; CHECK: %[[#R:]] = OpAtomicIAdd %[[#ulong]] %[[#]] %[[#scope_dev]] %[[#mem_uniform]] %[[#]]
  %r = call i64 @llvm.spv.interlocked.add.i64.p11(ptr addrspace(11) @dev_i64, i64 %v)
  ret i64 %r
}

declare i64 @llvm.spv.interlocked.add.i64.p3(ptr addrspace(3), i64)
declare i64 @llvm.spv.interlocked.add.i64.p11(ptr addrspace(11), i64)
