; RUN: opt -S -verify-each -passes='function(strip-convergence-intrinsics,spirv-legalize-resource-pointer-phis)' -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"Buf\00"

; The deoptimization state belongs to one predecessor and must stay there.
; CHECK-LABEL: define i32 @getpointer_bundle(
; CHECK:       a:
; CHECK-NEXT:    %idx = add i32 %i, 1
; CHECK-NEXT:    %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}} [ "deopt"(i32 %idx) ]
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %p, align 4
define i32 @getpointer_bundle(i1 %c, i32 %i) {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  br i1 %c, label %a, label %b
a:
  %idx = add i32 %i, 1
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 %idx) [ "deopt"(i32 %idx) ]
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)
