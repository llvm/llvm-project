; Test: Vulkan compute shader buffer access uses OpAccessChain (logical addressing)
; instead of OpPtrAccessChain (physical addressing).
;
; Before the fix in SPIRVSubtarget.h, isLogicalSPIRV() returned false for
; Vulkan shaders, so the backend generated OpPtrAccessChain (OpenCL style).
; After the fix, isLogicalSPIRV() returns true when isShader() is true,
; so the backend correctly uses OpAccessChain (Vulkan/GLSL style).
;
; Also verifies that the VulkanBuffer element type is wrapped in a Block struct
; (fixed in SPIRVGlobalRegistry.cpp). The buffer descriptor must be:
;   OpTypeStruct { OpTypeRuntimeArray { OpTypeStruct { float } } }
; not just a raw pointer to float.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#FLOAT:]]    = OpTypeFloat 32
; --- VulkanBuffer must wrap element in a struct (Block), not expose raw float ---
; CHECK-DAG: %[[#ELEM_WRAP:]] = OpTypeStruct %[[#FLOAT]]
; CHECK-DAG: %[[#RTARR:]]    = OpTypeRuntimeArray %[[#ELEM_WRAP]]
; CHECK-DAG: %[[#BUF_TY:]]   = OpTypeStruct %[[#RTARR]]

; --- Access must use OpAccessChain, NOT OpPtrAccessChain ---
; CHECK: OpAccessChain
; CHECK-NOT: OpPtrAccessChain

%struct.elem = type { float }

define void @main() #0 {
entry:
  ; Bind a VulkanBuffer of floats at set=0, binding=0
  %buf = call target("spirv.VulkanBuffer", [0 x %struct.elem], 12, 0)
      @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.elems_12_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; Get pointer to element [0] of the buffer
  %ptr = call noundef align 4 dereferenceable(4) ptr addrspace(11)
      @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.elems_12_0t(
          target("spirv.VulkanBuffer", [0 x %struct.elem], 12, 0) %buf, i32 0)

  ; Load the float value
  %val = load float, ptr addrspace(11) %ptr, align 4

  ; Store it back (write back test)
  store float %val, ptr addrspace(11) %ptr, align 4

  ret void
}

declare target("spirv.VulkanBuffer", [0 x %struct.elem], 12, 0)
    @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.elems_12_0t(i32, i32, i32, i32, ptr)

declare ptr addrspace(11)
    @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.elems_12_0t(
        target("spirv.VulkanBuffer", [0 x %struct.elem], 12, 0), i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
