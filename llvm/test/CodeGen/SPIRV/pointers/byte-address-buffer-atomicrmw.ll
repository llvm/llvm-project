; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute \
; RUN:   --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute \
; RUN:   --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | \
; RUN:   spirv-val %}

; Check that an atomic operation on a byte-address buffer uses an untyped
; resource pointer without an unsupported pointer cast.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"
; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#UINT]] 1
; CHECK: %[[#ATOMIC_PTR:]] = OpUntypedAccessChainKHR
; CHECK: OpAtomicAnd %[[#UINT]] %[[#ATOMIC_PTR]] {{%[0-9]+}} {{%[0-9]+}}
; CHECK-SAME: %[[#ONE]]

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

define void @main() #0 {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 1)
      @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0,
                                           ptr nonnull @.str)
  %ptr = tail call noundef align 4 dereferenceable(4) ptr addrspace(11)
      @llvm.spv.resource.getpointer(
          target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 0)
  %old = atomicrmw and ptr addrspace(11) %ptr, i32 1
      syncscope("device") monotonic, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

declare target("spirv.VulkanBuffer", [0 x i8], 12, 1)
    @llvm.spv.resource.handlefrombinding(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer(
    target("spirv.VulkanBuffer", [0 x i8], 12, 1), i32)
