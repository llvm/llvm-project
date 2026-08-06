; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Default (system) sync scope must not lower to CrossDevice on Vulkan targets.
; Monotonic atomics on storage-class memory also need AcquireRelease semantics.

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Scope_Device:]] = OpConstant %[[#Int]] 1{{$}}
; CHECK-DAG: %[[#MemSem:]] = OpConstant %[[#Int]] 72{{$}}
; CHECK-NOT: OpConstantNull %[[#Int]]

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

declare target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0), i32)

define void @main() #0 {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  atomicrmw or ptr addrspace(11) %ptr, i32 42 monotonic, align 4
  ; CHECK: OpAtomicOr {{.*}} %[[#Scope_Device:]] %[[#MemSem:]]
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
