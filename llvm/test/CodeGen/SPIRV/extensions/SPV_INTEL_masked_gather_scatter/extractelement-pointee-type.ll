; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; An element of a vector of pointers is extracted with the vector's component
; type and cast to the pointee type its user needs.
; CHECK-DAG: [[I8:%.+]] = OpTypeInt 8 0
; CHECK-DAG: [[PTR:%.+]] = OpTypePointer Generic [[I8]]
; CHECK-DAG: [[VEC:%.+]] = OpTypeVector [[PTR]] 4
; CHECK-DAG: [[ARR:%.+]] = OpTypeArray [[I8]] {{%.+}}
; CHECK-DAG: [[ARRPTR:%.+]] = OpTypePointer Generic [[ARR]]
; CHECK-DAG: [[CWPTR:%.+]] = OpTypePointer CrossWorkgroup [[I8]]
; CHECK-DAG: [[CWARRPTR:%.+]] = OpTypePointer CrossWorkgroup [[ARR]]
; CHECK-DAG: [[WGPTR:%.+]] = OpTypePointer Workgroup [[I8]]
; CHECK-DAG: [[WGARRPTR:%.+]] = OpTypePointer Workgroup [[ARR]]
; CHECK-DAG: [[FNPTR:%.+]] = OpTypePointer Function [[I8]]
; CHECK-DAG: [[FNARRPTR:%.+]] = OpTypePointer Function [[ARR]]
; CHECK: [[V:%.+]] = OpFunctionParameter [[VEC]]
; CHECK: [[E:%.+]] = OpCompositeExtract [[PTR]] [[V]] 0
; CHECK: [[C:%.+]] = OpBitcast [[ARRPTR]] [[E]]
; CHECK: OpPtrAccessChain [[ARRPTR]] [[C]]
define ptr addrspace(4) @f(<4 x ptr addrspace(4)> %v, i64 %i) {
  %p = extractelement <4 x ptr addrspace(4)> %v, i64 0
  %q = getelementptr [4 x i8], ptr addrspace(4) %p, i64 %i
  ret ptr addrspace(4) %q
}

; The same in other storage classes.
; CHECK: [[E:%.+]] = OpCompositeExtract [[CWPTR]] {{%.+}} 1
; CHECK: [[C:%.+]] = OpBitcast [[CWARRPTR]] [[E]]
; CHECK: OpPtrAccessChain [[CWARRPTR]] [[C]]
define ptr addrspace(1) @cross_workgroup(<4 x ptr addrspace(1)> %v, i64 %i) {
  %p = extractelement <4 x ptr addrspace(1)> %v, i64 1
  %q = getelementptr [4 x i8], ptr addrspace(1) %p, i64 %i
  ret ptr addrspace(1) %q
}

; CHECK: [[E:%.+]] = OpCompositeExtract [[WGPTR]] {{%.+}} 2
; CHECK: [[C:%.+]] = OpBitcast [[WGARRPTR]] [[E]]
; CHECK: OpPtrAccessChain [[WGARRPTR]] [[C]]
define ptr addrspace(3) @workgroup(<4 x ptr addrspace(3)> %v, i64 %i) {
  %p = extractelement <4 x ptr addrspace(3)> %v, i64 2
  %q = getelementptr [4 x i8], ptr addrspace(3) %p, i64 %i
  ret ptr addrspace(3) %q
}

; CHECK: [[E:%.+]] = OpCompositeExtract [[FNPTR]] {{%.+}} 3
; CHECK: [[C:%.+]] = OpBitcast [[FNARRPTR]] [[E]]
; CHECK: OpPtrAccessChain [[FNARRPTR]] [[C]]
define ptr @function(<4 x ptr> %v, i64 %i) {
  %p = extractelement <4 x ptr> %v, i64 3
  %q = getelementptr [4 x i8], ptr %p, i64 %i
  ret ptr %q
}
