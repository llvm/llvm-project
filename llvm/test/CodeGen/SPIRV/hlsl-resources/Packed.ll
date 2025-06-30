; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"


@.str.unpacked = private unnamed_addr constant [12 x i8] c"UnpackedRes\00", align 1
@.str.packed = private unnamed_addr constant [10 x i8] c"PackedRes\00", align 1

; CHECK-DAG: OpName [[unpacked:%[0-9]+]] "unpacked"
; CHECK-DAG: OpName [[packed:%[0-9]+]] "packed"

; CHECK-NOT: OpDecorate {{.*}} CPacked
; CHECK-DAG: OpMemberDecorate [[unpacked]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate [[unpacked]] 1 Offset 16

; CHECK-NOT: OpDecorate {{.*}} CPacked
; CHECK-DAG: OpMemberDecorate [[packed]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate [[packed]] 1 Offset 4
; CHECK-NOT: OpDecorate {{.*}} CPacked


%unpacked = type {i32, <3 x i32>}
%packed = type <{i32, <3 x i32>}>


define external i32 @unpacked_vulkan_buffer_load() {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x %unpacked], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str.unpacked)
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x %unpacked], 12, 0) %handle, i32 1)
  %1 = load i32, ptr addrspace(11) %0, align 4
  ret i32 %1
}

define external i32 @packed_vulkan_buffer_load() {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x %packed], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, i1 false, ptr nonnull @.str.packed)
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x %packed], 12, 0) %handle, i32 1)
  %1 = load i32, ptr addrspace(11) %0, align 4
  ret i32 %1
}
