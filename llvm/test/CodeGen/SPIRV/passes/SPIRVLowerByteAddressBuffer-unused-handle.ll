; RUN: opt -S -mtriple=spirv1.6-unknown-vulkan1.3-library -passes=spirv-lower-byte-address-buffer,verify %s | FileCheck %s --implicit-check-not='call target("spirv.VulkanBuffer", [0 x i8]'

; Assuming untyped pointers aren't supported, we should emit a i32[] handle for the
; raw buffer, and remove the dead i8[] handle

@name = private unnamed_addr constant [4 x i8] c"buf\00"

; CHECK-LABEL: define i32 @load_word(i32 %offset)
; CHECK: %[[ALIAS:.*]] = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding
; CHECK: %[[INDEX:.*]] = lshr i32 %offset, 2
; CHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %[[ALIAS]], i32 %[[INDEX]])
; CHECK: %[[VALUE:.*]] = load i32, ptr addrspace(11) %[[PTR]], align 4
; CHECK: ret i32 %[[VALUE]]
define i32 @load_word(i32 %offset) {
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @name)
  %value = call i32 @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 %offset, i32 4)
  ret i32 %value
}
