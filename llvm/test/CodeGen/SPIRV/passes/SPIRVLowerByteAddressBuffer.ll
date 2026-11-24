; RUN: opt -S -mtriple=spirv1.6-unknown-vulkan1.3-library -passes=spirv-lower-resource-names,spirv-lower-byte-address-buffer,verify %s | FileCheck %s

@name = private unnamed_addr constant [4 x i8] c"buf\00"

; The byte address buffer lowering pass should transform these load/store intrinsics
; into getpointer calls and llvm load/stores with appropriate alignment.
; Assumes untyped pointers are not supported, which necessitates creating
; aliasing reinterpretations (e.g. uint[]) of the original resource handle

; CHECK-LABEL: define i32 @scalar(i32 %offset)
; CHECK-DAG: %[[INDEX:.*]] = lshr i32 %offset, 2
; CHECK-DAG: %[[H:.*]] = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname
; CHECK: %[[P:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %[[H]], i32 %[[INDEX]])
; CHECK: %[[V:.*]] = load i32, ptr addrspace(11) %[[P]], align 4
; CHECK: store i32 %[[V]], ptr addrspace(11) %{{.*}}, align 4
; CHECK: ret i32 %[[V]]
define i32 @scalar(i32 %offset) {
  %h = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr @name)
  %v = call i32 @llvm.spv.resource.load.rawbuffer.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 %offset, i32 4)
  call void @llvm.spv.resource.store.rawbuffer.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 %offset, i32 %v, i32 4)
  ret i32 %v
}

; CHECK-LABEL: define <8 x i32> @vector()
; CHECK: call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}({{.*}}, i32 1)
; CHECK-COUNT-8: load i32, ptr addrspace(11) %{{.*}}, align 4
; CHECK-COUNT-8: store i32 %{{.*}}, ptr addrspace(11) %{{.*}}, align 4
; CHECK: ret <8 x i32>
define <8 x i32> @vector() {
  %h = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr @name)
  %v = call <8 x i32> @llvm.spv.resource.load.rawbuffer.v8i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 4, i32 4)
  call void @llvm.spv.resource.store.rawbuffer.v8i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 4, <8 x i32> %v, i32 4)
  ret <8 x i32> %v
}

; CHECK-LABEL: define <3 x float> @aligned_vec3()
; CHECK-COUNT-3: load float, ptr addrspace(11) %{{.*}}, align {{[0-9]+}}
; CHECK: ret <3 x float>
define <3 x float> @aligned_vec3() {
  %h = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr @name)
  %v = call <3 x float> @llvm.spv.resource.load.rawbuffer.v3f32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 0, i32 16)
  ret <3 x float> %v
}

; alignment is only 1 so we shouldn't emit a i32 access with align 4, rather leave for
; downstream legalization

; CHECK-LABEL: define i32 @weak_alignment()
; CHECK: %[[BYTEPTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 4)
; CHECK: load i32, ptr addrspace(11) %[[BYTEPTR]], align 1
define i32 @weak_alignment() {
  %h = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr @name)
  %v = call i32 @llvm.spv.resource.load.rawbuffer.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 4, i32 1)
  ret i32 %v
}

; Insert new aliased handles before the original handle, so operands are available
; and handle uses are dominated
; CHECK-LABEL: define i32 @reuse_across_blocks(i1 %condition, i32 %index)
; CHECK: %[[NEXTINDEX:.*]] = add i32 %index, 1
; CHECK-NEXT: %[[ALIAS0:.*]] = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 3, i32 4, i32 %index,
; CHECK-NEXT: %[[ALIAS1:.*]] = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 3, i32 4, i32 %[[NEXTINDEX]],
; CHECK-NEXT: br i1 %condition
; CHECK: left:
; CHECK: @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %[[ALIAS0]], i32 0)
; CHECK: right:
; CHECK: @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %[[ALIAS0]], i32 1)
; CHECK: join:
; CHECK: @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %[[ALIAS1]], i32 0)
; CHECK: ret i32
define i32 @reuse_across_blocks(i1 %condition, i32 %index) {
entry:
  %nextindex = add i32 %index, 1
  %h0 = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 4, i32 %index, ptr @name)
  %h1 = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 4, i32 %nextindex, ptr @name)
  br i1 %condition, label %left, label %right
left:
  %a = call i32 @llvm.spv.resource.load.rawbuffer.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h0, i32 0, i32 4)
  br label %join
right:
  %b = call i32 @llvm.spv.resource.load.rawbuffer.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h0, i32 4, i32 4)
  br label %join
join:
  %v = phi i32 [ %a, %left ], [ %b, %right ]
  call void @llvm.spv.resource.store.rawbuffer.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h1, i32 0, i32 %v, i32 4)
  ret i32 %v
}

; Distinct accessed types using the same handle need separate aliases
; CHECK-LABEL: define float @different_alias_types(i32 %index)
; CHECK: %[[INTALIAS:.*]] = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname
; CHECK-NEXT: %[[FLOATALIAS:.*]] = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.mdname
; CHECK: @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %[[INTALIAS]], i32 0)
; CHECK: @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x float], 12, 1) %[[FLOATALIAS]], i32 1)
; CHECK: ret float
define float @different_alias_types(i32 %index) {
  %h = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 4, i32 %index, ptr @name)
  %i = call i32 @llvm.spv.resource.load.rawbuffer.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 0, i32 4)
  %f = call float @llvm.spv.resource.load.rawbuffer.f32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %h, i32 4, i32 4)
  %converted = uitofp i32 %i to float
  %sum = fadd float %f, %converted
  ret float %sum
}

; CHECK: !{!".resource.alias.0.3.i32.rw.buf"}
