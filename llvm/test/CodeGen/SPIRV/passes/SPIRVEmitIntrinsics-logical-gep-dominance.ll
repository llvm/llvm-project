; RUN: opt -S -passes=spirv-emit-intrinsics \
; RUN:   -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s

; Verify that the replacement llvm.spv.gep remains after its pointer operand.
; Regression test for https://github.com/llvm/llvm-project/issues/226608.

%struct.MaskStruct = type { [4 x i32] }

define void @logical_access_chain(
    target("spirv.VulkanBuffer", [0 x %struct.MaskStruct], 12, 0) %buffer,
    i32 %offset) {
entry:
  %index = zext i32 %offset to i64
  %base = call ptr addrspace(11)
      @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.MaskStructs_12_0t.i32(
          target("spirv.VulkanBuffer", [0 x %struct.MaskStruct], 12, 0) %buffer,
          i32 0)
  %element = getelementptr [4 x i8], ptr addrspace(11) %base, i64 %index
  store i32 0, ptr addrspace(11) %element
  ret void

; CHECK-LABEL: define void @logical_access_chain(
; CHECK: [[INDEX:%.*]] = zext i32 %offset to i64
; CHECK: [[BASE:%.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer
; CHECK: [[ELEMENT:%.*]] = call ptr addrspace(11)
; CHECK-SAME: @llvm.spv.gep.p11.p11(
; CHECK-SAME: ptr addrspace(11) [[BASE]]
; CHECK-SAME: i64 [[INDEX]])
; CHECK: store i32 0, ptr addrspace(11) [[ELEMENT]]
}

declare ptr addrspace(11)
    @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.MaskStructs_12_0t.i32(
        target("spirv.VulkanBuffer", [0 x %struct.MaskStruct], 12, 0), i32)
