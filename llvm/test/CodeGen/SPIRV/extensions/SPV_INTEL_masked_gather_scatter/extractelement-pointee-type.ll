; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; An element of a vector of pointers is extracted with the vector's component
; type and cast to the pointee type its user needs.
; CHECK-DAG: [[I8:%.+]] = OpTypeInt 8 0
; CHECK-DAG: [[PTR:%.+]] = OpTypePointer Generic [[I8]]
; CHECK-DAG: [[VEC:%.+]] = OpTypeVector [[PTR]] 4
; CHECK-DAG: [[ARR:%.+]] = OpTypeArray [[I8]] {{%.+}}
; CHECK-DAG: [[ARRPTR:%.+]] = OpTypePointer Generic [[ARR]]
; CHECK: [[V:%.+]] = OpFunctionParameter [[VEC]]
; CHECK: [[E:%.+]] = OpCompositeExtract [[PTR]] [[V]] 0
; CHECK: [[C:%.+]] = OpBitcast [[ARRPTR]] [[E]]
; CHECK: OpPtrAccessChain [[ARRPTR]] [[C]]
define ptr addrspace(4) @f(<4 x ptr addrspace(4)> %v, i64 %i) {
  %p = extractelement <4 x ptr addrspace(4)> %v, i64 0
  %q = getelementptr [4 x i8], ptr addrspace(4) %p, i64 %i
  ret ptr addrspace(4) %q
}
