; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:       %[[#float:]] = OpTypeFloat 32
; CHECK-DAG:        %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:     %[[#v2_uint:]] = OpTypeVector %[[#uint]] 2
; CHECK-DAG:      %[[#double:]] = OpTypeFloat 64
; CHECK-DAG:   %[[#v2_double:]] = OpTypeVector %[[#double]] 2
; CHECK-DAG:    %[[#v4_float:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG:     %[[#v4_uint:]] = OpTypeVector %[[#uint]] 4
; CHECK-DAG:       %[[#ulong:]] = OpTypeInt 64 0
; CHECK-DAG:    %[[#v2_ulong:]] = OpTypeVector %[[#ulong]] 2
; CHECK-DAG:     %[[#v3_uint:]] = OpTypeVector %[[#uint]] 3
@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Out\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %in_buffer_handle = tail call target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %out_buffer_handle = tail call target("spirv.VulkanBuffer", [0 x <2 x double>], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.2)
  %src0_ptr = tail call noundef align 8 dereferenceable(8) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) %in_buffer_handle, i32 0)
  %src0 = load <2 x i32>, ptr addrspace(11) %src0_ptr, align 8
  %src1_ptr = tail call noundef align 8 dereferenceable(8) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) %in_buffer_handle, i32 1)
  %src1 = load <2 x i32>, ptr addrspace(11) %src1_ptr, align 8
; CHECK: %[[#tmp:]] = OpVectorShuffle %[[#v4_uint]] {{%[0-9]+}} {{%[0-9]+}} 0 2 1 3
  %shuffled = shufflevector <2 x i32> %src0, <2 x i32> %src1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
; CHECK: %[[#access:]] = OpAccessChain {{.*}}
  %dst_ptr = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <2 x double>], 12, 1) %out_buffer_handle, i32 0)
; CHECK: %[[#bitcast:]] = OpBitcast %[[#v2_double]] %[[#tmp]]
; CHECK: OpStore %[[#access]] %[[#bitcast]]
  store <4 x i32> %shuffled, ptr addrspace(11) %dst_ptr, align 16
  ret void
}

; This tests a load from a pointer that has been bitcast between vector types
; which share the same total bit-width but have different numbers of elements.
; Tests that legalize-pointer-casts works correctly by moving the bitcast to
; the element that was loaded.

define void @main2() local_unnamed_addr #0 {
entry:
; CHECK:  %[[LOAD:[0-9]+]] = OpLoad %[[#v2_double]] {{.*}}
; CHECK:  %[[BITCAST1:[0-9]+]] = OpBitcast %[[#v4_uint]] %[[LOAD]]
; CHECK:  %[[BITCAST2:[0-9]+]] = OpBitcast %[[#v2_double]] %[[BITCAST1]]
; CHECK: OpStore {{%[0-9]+}} %[[BITCAST2]] {{.*}}

  %out_buffer_handle = tail call target("spirv.VulkanBuffer", [0 x <2 x double>], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.2)
  %src_ptr = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <2 x double>], 12, 1) %out_buffer_handle, i32 0)
  %loaded_v4i32 = load <4 x i32>, ptr addrspace(11) %src_ptr
  %dst_ptr = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <2 x double>], 12, 1) %out_buffer_handle, i32 1)
  store <4 x i32> %loaded_v4i32, ptr addrspace(11) %dst_ptr
  ret void
}

@.str.3 = private unnamed_addr constant [4 x i8] c"In2\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"Out2\00", align 1

define void @main3() local_unnamed_addr #0 {
entry:
; CHECK:       %[[LOAD3:[0-9]+]] = OpLoad %[[#v4_float]] {{.*}}
; CHECK-NEXT:  %[[BITCAST3:[0-9]+]] = OpBitcast %[[#v4_uint]] %[[LOAD3]]
; CHECK-NEXT:  %[[SHUFFLE3:[0-9]+]] = OpVectorShuffle %[[#v2_uint]] %[[BITCAST3]] %[[BITCAST3]] 0 1
; CHECK:       OpStore {{%[0-9]+}} %[[SHUFFLE3]] {{.*}}

  %in_buffer_handle = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 0) @llvm.spv.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr nonnull @.str.3)
  %out_buffer_handle = tail call target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) @llvm.spv.resource.handlefrombinding(i32 1, i32 1, i32 1, i32 0, ptr nonnull @.str.4)
  %src_ptr = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 0) %in_buffer_handle, i32 0)
  %loaded_v2i32 = load <2 x i32>, ptr addrspace(11) %src_ptr, align 16
  %dst_ptr = tail call noundef align 8 dereferenceable(8) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) %out_buffer_handle, i32 0)
  store <2 x i32> %loaded_v2i32, ptr addrspace(11) %dst_ptr, align 8
  ret void
}

; Tests loading a vector where the source total bit width is not evenly
; divisible by the target element bit width.

@.str.in = private unnamed_addr constant [4 x i8] c"In3\00", align 1
@.str.out = private unnamed_addr constant [5 x i8] c"Out3\00", align 1

define void @main4() local_unnamed_addr #0 {
entry:
; CHECK:       %[[LOAD:[0-9]+]] = OpLoad %[[#v3_uint]] {{.*}}
; CHECK-NEXT:  %[[SHUFFLE:[0-9]+]] = OpVectorShuffle %[[#v4_uint]] %[[LOAD]] %[[LOAD]] 0 1 2 0xFFFFFFFF{{.*}}
; CHECK-NEXT:  %[[BITCAST:[0-9]+]] = OpBitcast %[[#v2_ulong]] %[[SHUFFLE]]{{.*}}
; CHECK:       OpStore {{%[0-9]+}} %[[BITCAST]] {{.*}}

  %in_buffer_handle = tail call target("spirv.VulkanBuffer", [0 x <3 x i32>], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str.in)
  %out_buffer_handle = tail call target("spirv.VulkanBuffer", [0 x <2 x i64>], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.out)
  %src_ptr = tail call noundef align 16 dereferenceable(12) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <3 x i32>], 12, 0) %in_buffer_handle, i32 0)
  %loaded_v2i64 = load <2 x i64>, ptr addrspace(11) %src_ptr, align 16
  %dst_ptr = tail call noundef align 8 dereferenceable(8) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <2 x i64>], 12, 0) %out_buffer_handle, i32 0)
  store <2 x i64> %loaded_v2i64, ptr addrspace(11) %dst_ptr, align 8
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
