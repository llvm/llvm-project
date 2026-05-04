; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; Test lowering of llvm.spv.interlocked.add to OpAtomicIAdd in workgroup storage.

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#scope_wg:]] = OpConstant %[[#uint]] 2
; CHECK-DAG: %[[#mem_wg:]] = OpConstant %[[#uint]] 256

@gs_i32 = internal addrspace(3) global i32 zeroinitializer

; CHECK-LABEL: Begin function test_i32
define i32 @test_i32(i32 %v) {
entry:
; CHECK: %[[#R:]] = OpAtomicIAdd %[[#uint]] %[[#]] %[[#scope_wg]] %[[#mem_wg]] %[[#]]
  %r = call i32 @llvm.spv.interlocked.add.i32.p3(ptr addrspace(3) @gs_i32, i32 %v)
  ret i32 %r
}

declare i32 @llvm.spv.interlocked.add.i32.p3(ptr addrspace(3), i32)
