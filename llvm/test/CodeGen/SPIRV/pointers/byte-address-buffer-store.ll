; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[AC0:%[0-9]+]] = OpAccessChain {{.*}} %13 %13
; CHECK: OpStore [[AC0]]
; CHECK-DAG: [[AC1:%[0-9]+]] = OpAccessChain {{.*}} %13 %17
; CHECK: OpStore [[AC1]]
; CHECK-DAG: [[AC2:%[0-9]+]] = OpAccessChain {{.*}} %13 %15
; CHECK: OpStore [[AC2]]
; CHECK-DAG: [[AC3:%[0-9]+]] = OpAccessChain {{.*}} %13 %14
; CHECK: OpStore [[AC3]]

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store i32 42, ptr addrspace(11) %ptr, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
