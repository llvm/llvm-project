; RUN: llc -O0 -mtriple=spirv-vulkan-library %s -o - | FileCheck %s --implicit-check-not="OpCapability VariablePointers"
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -O0 -mtriple=spirv-vulkan-library -enable-new-pm %s -o - | FileCheck %s --implicit-check-not="OpCapability VariablePointers"
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library -enable-new-pm %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; A phi or select of resource pointers on one handle should become a phi or
; select of the element index, used by the image or buffer access directly,
; including a struct member load and a storage-buffer atomic add.

%struct.S = type { i32, i32 }

@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"Out\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Buf\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"Structs\00", align 1

; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0

; CHECK-LABEL: ; -- Begin function phi_texel_load{{$}}
; CHECK: %[[#LOAD_IDX:]] = OpPhi %[[#UINT]]
; CHECK: OpImageRead %[[#]] %[[#]] %[[#LOAD_IDX]]
define void @phi_texel_load() #0 {
entry:
  %in = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %out = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %c = icmp eq i32 %tid, 0
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %in, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %in, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %v = load i32, ptr addrspace(11) %p, align 4
  %q = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %out, i32 %tid)
  store i32 %v, ptr addrspace(11) %q, align 4
  ret void
}

; CHECK-LABEL: ; -- Begin function phi_texel_store{{$}}
; CHECK: %[[#STORE_IDX:]] = OpPhi %[[#UINT]]
; CHECK: OpImageWrite %[[#]] %[[#STORE_IDX]]
define void @phi_texel_store() #0 {
entry:
  %out = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %c = icmp eq i32 %tid, 0
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %out, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %out, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  store i32 %tid, ptr addrspace(11) %p, align 4
  ret void
}

; CHECK-LABEL: ; -- Begin function select_storage_buffer{{$}}
; CHECK: %[[#SEL_IDX:]] = OpSelect %[[#UINT]]
; CHECK: %[[#PTR:]] = OpAccessChain %[[#]] %[[#]] %[[#]] %[[#SEL_IDX]]
; CHECK: OpLoad %[[#UINT]] %[[#PTR]]
define void @select_storage_buffer() #0 {
entry:
  %buf = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %c = icmp eq i32 %tid, 0
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %buf, i32 0)
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %buf, i32 1)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1
  %v = load i32, ptr addrspace(11) %p, align 4
  %q = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %buf, i32 %tid)
  store i32 %v, ptr addrspace(11) %q, align 4
  ret void
}

; CHECK-LABEL: ; -- Begin function phi_struct_member{{$}}
; CHECK: %[[#MEMBER_IDX:]] = OpPhi %[[#UINT]]
; CHECK: %[[#ELEM:]] = OpAccessChain %[[#]] %[[#]] %[[#]] %[[#MEMBER_IDX]]
; CHECK: %[[#MEMBER:]] = {{OpInBoundsAccessChain|OpAccessChain}} %[[#]] %[[#ELEM]] %[[#]]
; CHECK: OpLoad %[[#UINT]] %[[#MEMBER]]
define void @phi_struct_member() #0 {
entry:
  %buf = call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(i32 0, i32 3, i32 1, i32 0, ptr @.str.3)
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %c = icmp eq i32 %tid, 0
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %buf, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %buf, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %m = getelementptr inbounds nuw i8, ptr addrspace(11) %p, i64 4
  %v = load i32, ptr addrspace(11) %m, align 4
  %q = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %buf, i32 %tid)
  store i32 %v, ptr addrspace(11) %q, align 4
  ret void
}

; A switch inside the loop selects the texel index used by the read.
; CHECK-LABEL: ; -- Begin function phi_texel_loop{{$}}
; CHECK: OpLoopMerge
; CHECK: %[[#CASE_IDX:]] = OpPhi %[[#UINT]]
; CHECK: OpImageRead %[[#]] %[[#]] %[[#CASE_IDX]]
; CHECK: OpFunctionEnd
define void @phi_texel_loop() #0 {
entry:
  %ce = call token @llvm.experimental.convergence.entry()
  %in = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %out = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %read ]
  %cl = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %ce) ]
  switch i32 %i, label %unreachable [
    i32 0, label %a
    i32 1, label %b
    i32 2, label %exit
  ]
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %in, i32 3) [ "convergencectrl"(token %cl) ]
  br label %read
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %in, i32 7) [ "convergencectrl"(token %cl) ]
  br label %read
read:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %v = load i32, ptr addrspace(11) %p, align 4
  %q = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %out, i32 %i) [ "convergencectrl"(token %cl) ]
  store i32 %v, ptr addrspace(11) %q, align 4
  %next = add i32 %i, 1
  br label %loop
unreachable:
  unreachable
exit:
  ret void
}

; CHECK-LABEL: ; -- Begin function phi_storage_buffer_atomicrmw{{$}}
; CHECK: %[[#ATOMIC_IDX:]] = OpPhi %[[#UINT]]
; CHECK: %[[#ATOMIC_PTR:]] = OpAccessChain %[[#]] %[[#]] %[[#]] %[[#ATOMIC_IDX]]
; CHECK: OpAtomicIAdd %[[#UINT]] %[[#ATOMIC_PTR]]
define void @phi_storage_buffer_atomicrmw() #0 {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %c = icmp eq i32 %tid, 0
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 2)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %old = atomicrmw add ptr addrspace(11) %p, i32 1 syncscope("device") monotonic, align 4
  ret void
}

declare target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24), i32)
declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)
declare target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1), i32)
declare i32 @llvm.spv.thread.id.i32(i32)
declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.loop()

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
