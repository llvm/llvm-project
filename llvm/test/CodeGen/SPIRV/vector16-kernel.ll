; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Kernels have the Vector16 capability, so 8 and 16 element vectors should use plain
; OpTypeVector whether SPV_EXT_long_vector is enabled or not.

; CHECK-DAG: OpCapability Vector16
; CHECK-NOT: OpCapability LongVectorEXT
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V8:]] = OpTypeVector %[[#FLOAT]] 8
; CHECK-DAG: %[[#V16:]] = OpTypeVector %[[#FLOAT]] 16
; CHECK-NOT: OpTypeVectorIdEXT

define spir_kernel void @k(<8 x float> %a, <16 x float> %b, ptr addrspace(1) %o8, ptr addrspace(1) %o16) {
  ; CHECK: OpFAdd %[[#V8]]
  %x = fadd <8 x float> %a, %a
  ; CHECK: OpFAdd %[[#V16]]
  %y = fadd <16 x float> %b, %b
  store <8 x float> %x, ptr addrspace(1) %o8
  store <16 x float> %y, ptr addrspace(1) %o16
  ret void
}
