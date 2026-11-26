; RUN: opt -S -verify-each -passes='function(spirv-legalize-resource-pointer-phis)' -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"Buf\00"

; A catchswitch block has room for phis but cannot contain a call.
; CHECK-LABEL: define void @catchswitch_phi(
; CHECK:       dispatch:
; CHECK-NEXT:    %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
; CHECK-NEXT:    %cs = catchswitch within none [label %catch] unwind to caller
; CHECK:       catch:
; CHECK-NEXT:    %pad = catchpad within %cs []
; CHECK-NEXT:    store i32 1, ptr addrspace(11) %p, align 4
define void @catchswitch_phi(i1 %c) personality ptr @__CxxFrameHandler3 {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  invoke void @may_throw() to label %exit unwind label %dispatch
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1)
  invoke void @may_throw() to label %exit unwind label %dispatch
dispatch:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %pad = catchpad within %cs []
  store i32 1, ptr addrspace(11) %p, align 4
  catchret from %pad to label %exit
exit:
  ret void
}

declare i32 @__CxxFrameHandler3(...)
declare void @may_throw()
declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)
