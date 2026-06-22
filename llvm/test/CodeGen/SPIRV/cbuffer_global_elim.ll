; RUN: opt -S -passes='spirv-cbuffer-access' %s -o - | FileCheck %s
; RUN: llc %s -o - | FileCheck %s
; RUN: llc %s -O3 -o - | FileCheck %s

target triple = "spirv-unknown-vulkan1.3-compute"

%__cblayout_CB = type <{ i32 }>
@CB.cb = internal global target("spirv.VulkanBuffer", %__cblayout_CB, 2, 0) poison
@i = external hidden local_unnamed_addr addrspace(2) global i32, align 4

@llvm.compiler.used = appending global [1 x ptr] [ptr @CB.cb], section "llvm.metadata"

; Check that SPRIVCBufferAccessPass removes the cbuffer global from @llvm.compiler.used
; and from the module.
;
; CHECK-NOT: @CB.cb = internal global target("spirv.VulkanBuffer", %__cblayout_CB, 2, 0) poison
; CHECK-NOT: @i
; CHECK-NOT: @llvm.compiler.used

; Check that the cbuffer global is removed during lowering to DXIL
;
define void @main() {
entry:
  %cb_handle = tail call target("spirv.VulkanBuffer", %__cblayout_CB, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_CBs_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull null)
  store target("spirv.VulkanBuffer", %__cblayout_CB, 2, 0) %cb_handle, ptr @CB.cb, align 4
  %1 = load i32, ptr addrspace(2) @i, align 4
  ret void
}

; CHECK-NOT: !hlsl.cbs

!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @i}
!1 = !{i32 1, i32 8}

