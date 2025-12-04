; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 -stop-after=prologepilog < %s | FileCheck -check-prefix=CHECK %s

; Make sure we use a stack pointer and allocate 112 * 4 bytes at the beginning of the stack.

define amdgpu_cs void @amdgpu_cs() #0 {
; CHECK-LABEL: {{^}}name: amdgpu_cs
; CHECK: scratchReservedForDynamicVGPRs: 448
  ret void
}

define amdgpu_kernel void @kernel() #0 {
; CHECK-LABEL: {{^}}name: kernel
; CHECK: scratchReservedForDynamicVGPRs: 448
  ret void
}

define amdgpu_cs void @with_local() #0 {
; CHECK-LABEL: {{^}}name: with_local
; CHECK: scratchReservedForDynamicVGPRs: 448
  %local = alloca i32, addrspace(5)
  store volatile i8 13, ptr addrspace(5) %local
  ret void
}

define amdgpu_cs void @with_calls() #0 {
; CHECK-LABEL: {{^}}name: with_calls
; CHECK: scratchReservedForDynamicVGPRs: 448
  %local = alloca i32, addrspace(5)
  store volatile i8 15, ptr addrspace(5) %local
  call amdgpu_gfx void @callee(i32 71)
  ret void
}

define amdgpu_cs void @realign_stack(<32 x i32> %x) #0 {
; CHECK-LABEL: {{^}}name: realign_stack
; CHECK: scratchReservedForDynamicVGPRs: 512
  %v = alloca <32 x i32>, align 128, addrspace(5)
  ; use volatile store to avoid promotion of alloca to registers
  store volatile <32 x i32> %x, ptr addrspace(5) %v
  call amdgpu_gfx void @callee(i32 71)
  ret void
}

; Non-entry functions and graphics shaders can't run on a compute queue,
; so they don't need to worry about CWSR.
define amdgpu_gs void @amdgpu_gs() #0 {
; CHECK-LABEL: {{^}}name: amdgpu_gs
; CHECK: scratchReservedForDynamicVGPRs: 0
  %local = alloca i32, addrspace(5)
  store volatile i8 15, ptr addrspace(5) %local
  call amdgpu_gfx void @callee(i32 71)
  ret void
}

define amdgpu_gfx void @amdgpu_gfx() #0 {
; CHECK-LABEL: {{^}}name: amdgpu_gfx
; CHECK: scratchReservedForDynamicVGPRs: 0
  %local = alloca i32, addrspace(5)
  store volatile i8 15, ptr addrspace(5) %local
  call amdgpu_gfx void @callee(i32 71)
  ret void
}

define void @default() #0 {
; CHECK-LABEL: {{^}}name: default
; CHECK: scratchReservedForDynamicVGPRs: 0
  ret void
}

declare amdgpu_gfx void @callee(i32) #0

attributes #0 = { nounwind "amdgpu-dynamic-vgpr-block-size" = "16" }

