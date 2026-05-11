; Verify that volatile atomic load/store are accepted and produce
; OpAtomicLoad/OpAtomicStore. On non-Vulkan targets the volatile qualifier
; is silently dropped; on Vulkan the Volatile memory semantics bit is set.

; Non-Vulkan: no Volatile memory semantics.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=NONVK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Vulkan: Volatile memory semantics (0x8000) ORed into the semantics operand.
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefix=VULKAN

; NONVK-DAG: %[[#C514:]] = OpConstant %[[#]] 514
; NONVK-DAG: %[[#C516:]] = OpConstant %[[#]] 516
; NONVK: OpAtomicLoad %[[#]] %[[#]] %[[#]] %[[#C514]]
; NONVK: OpAtomicStore %[[#]] %[[#]] %[[#C516]] %[[#]]

; 0x8000 | 0x2 (Acquire) | 0x200 (CrossWorkgroupMemory) = 33282
; 0x8000 | 0x4 (Release) | 0x200 (CrossWorkgroupMemory) = 33284
; VULKAN-DAG: %[[#VA:]] = OpConstant %[[#]] 33282
; VULKAN-DAG: %[[#VR:]] = OpConstant %[[#]] 33284
; VULKAN: OpAtomicLoad %[[#]] %[[#]] %[[#]] %[[#VA]]
; VULKAN: OpAtomicStore %[[#]] %[[#]] %[[#VR]] %[[#]]

define i32 @load_volatile(ptr addrspace(1) %ptr) {
  %val = load atomic volatile i32, ptr addrspace(1) %ptr syncscope("device") acquire, align 4
  ret i32 %val
}

define void @store_volatile(ptr addrspace(1) %ptr, i32 %val) {
  store atomic volatile i32 %val, ptr addrspace(1) %ptr syncscope("device") release, align 4
  ret void
}
