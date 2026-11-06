; Vulkan counterpart of atomicrmw-storage-class-semantics.ll.
;
; Vulkan forbids a storage-class MemorySemantics bit combined with a relaxed
; order (VUID-StandaloneSpirv-MemorySemantics-10871), so relaxed atomics must
; drop the bit instead of OR'ing it in. See
; atomiccmpxchg-storage-class-semantics-vulkan.ll for the cmpxchg case.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=CHECK,NO-SC-BIT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Workgroup:]] = OpConstant %[[#Int]] 2
; None (0) semantics is emitted as OpConstantNull.
; CHECK-DAG: %[[#None:]] = OpConstantNull %[[#Int]]

; NOT-only prefix scans the whole module, since the constant precedes the atomics in output.
; NO-SC-BIT-NOT: OpConstant %[[#Int]] 256

@gs = external hidden addrspace(3) global i32, align 4

define void @rmw() #0 {
  ; CHECK: OpAtomicOr %[[#Int]] %[[#]] %[[#Workgroup]] %[[#None]] %[[#]]
  %r = atomicrmw or ptr addrspace(3) @gs, i32 1 syncscope("workgroup") monotonic, align 4
  ret void
}

define void @ld() #0 {
  ; CHECK: OpAtomicLoad %[[#Int]] %[[#]] %[[#Workgroup]] %[[#None]]
  %v = load atomic i32, ptr addrspace(3) @gs syncscope("workgroup") monotonic, align 4
  ret void
}

define void @st() #0 {
  ; CHECK: OpAtomicStore %[[#]] %[[#Workgroup]] %[[#None]] %[[#]]
  store atomic i32 1, ptr addrspace(3) @gs syncscope("workgroup") monotonic, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
