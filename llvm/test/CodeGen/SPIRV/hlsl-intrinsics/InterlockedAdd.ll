; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; Test lowering of llvm.spv.interlocked.add to OpAtomicIAdd.

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#scope_wg:]] = OpConstant %[[#uint]] 2
; CHECK-DAG: %[[#scope_dev:]] = OpConstant %[[#uint]] 1
; CHECK-DAG: %[[#mem_wg:]] = OpConstant %[[#uint]] 256
; CHECK-DAG: %[[#mem_uniform:]] = OpConstant %[[#uint]] 64

@gs_i32 = internal addrspace(3) global i32 zeroinitializer
@dev_i32 = external addrspace(11) global i32

; Workgroup (addrspace 3) memory tests.

; CHECK-LABEL: Begin function test_i32
define i32 @test_i32(i32 %v) {
entry:
; CHECK: %[[#R:]] = OpAtomicIAdd %[[#uint]] %[[#]] %[[#scope_wg]] %[[#mem_wg]] %[[#]]
  %r = call i32 @llvm.spv.interlocked.add.i32.p3(ptr addrspace(3) @gs_i32, i32 %v)
  ret i32 %r
}

; Device / StorageBuffer (addrspace 11) memory tests.

; CHECK-LABEL: Begin function test_device_i32
define i32 @test_device_i32(i32 %v) {
entry:
; CHECK: %[[#R:]] = OpAtomicIAdd %[[#uint]] %[[#]] %[[#scope_dev]] %[[#mem_uniform]] %[[#]]
  %r = call i32 @llvm.spv.interlocked.add.i32.p11(ptr addrspace(11) @dev_i32, i32 %v)
  ret i32 %r
}

declare i32 @llvm.spv.interlocked.add.i32.p3(ptr addrspace(3), i32)
declare i32 @llvm.spv.interlocked.add.i32.p11(ptr addrspace(11), i32)
