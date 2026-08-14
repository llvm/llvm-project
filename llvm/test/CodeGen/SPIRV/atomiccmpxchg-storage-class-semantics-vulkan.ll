; Vulkan cmpxchg counterpart of atomicrmw-storage-class-semantics-vulkan.ll;
; see that file for the VUID this fixes.
;
; TODO: a Workgroup-storage-class cmpxchg crossing a function boundary (call
; arg or OpEntryPoint param) hits an unrelated crash in
; SPIRVLegalizePointerCast. Once fixed, add:
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Workgroup:]] = OpConstant %[[#Int]] 2

define spir_func void @cmpxchg(ptr addrspace(3) %p) {
  ; WorkgroupMemory bit (256) must not appear on either semantics operand.
  ; CHECK: OpAtomicCompareExchange %[[#Int]] %[[#]] %[[#Workgroup]] %[[#]] %[[#]] %[[#]] %[[#]]
  ; CHECK-NOT: OpConstant %[[#Int]] 256
  %pair = cmpxchg ptr addrspace(3) %p, i32 0, i32 1 syncscope("workgroup") monotonic monotonic
  ret void
}
