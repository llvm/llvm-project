; In a Vulkan (shader) environment a relaxed (monotonic) atomic must NOT set a
; storage-class memory-semantics bit (e.g. WorkgroupMemory): Vulkan requires
; such bits to be paired with a non-relaxed order (Acquire/Release/AcqRel), so
; pairing WorkgroupMemory with a relaxed order is rejected by spirv-val
; (VUID-StandaloneSpirv-MemorySemantics-10871). This is the shape emitted for
; HLSL Interlocked* on `groupshared` memory, which uses a "workgroup" syncscope
; and a relaxed order. The scope must still be Workgroup and the semantics None.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#U32:]] = OpTypeInt 32 0
; Workgroup scope = 2, None (relaxed) memory semantics = 0.
; CHECK-DAG: %[[#WG:]] = OpConstant %[[#U32]] 2
; CHECK-DAG: %[[#NONE:]] = OpConstantNull %[[#U32]]

@gs = external hidden addrspace(3) global i32, align 4

define void @main() #0 {
entry:
  ; CHECK: OpAtomicOr %[[#U32]] %{{[0-9]+}} %[[#WG]] %[[#NONE]] %{{[0-9]+}}
  %0 = atomicrmw or ptr addrspace(3) @gs, i32 1 syncscope("workgroup") monotonic
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
