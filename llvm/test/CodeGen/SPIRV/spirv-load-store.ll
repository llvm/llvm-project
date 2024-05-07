; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
;; Translate SPIR-V friendly OpLoad and OpStore calls

; CHECK-DAG: %[[#TYLONG:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TYFLOAT:]] = OpTypeFloat 64
; CHECK-DAG: %[[#TYFLOATPTR:]] = OpTypePointer CrossWorkgroup %[[#TYFLOAT]]
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#TYLONG]] 42
; CHECK: OpStore %[[#PTRTOLONG:]] %[[#CONST]] Volatile|Aligned 4
; CHECK: %[[#PTRTOFLOAT:]] = OpBitcast %[[#TYFLOATPTR]] %[[#PTRTOLONG]]
; CHECK: OpLoad %[[#TYFLOAT]] %[[#PTRTOFLOAT]]

define weak_odr dso_local spir_kernel void @foo(i32 addrspace(1)* %var) {
entry:
  tail call spir_func void @_Z13__spirv_StorePiiii(i32 addrspace(1)* %var, i32 42, i32 3, i32 4)
  %value = tail call spir_func double @_Z12__spirv_LoadPi(i32 addrspace(1)* %var)
  ret void
}

declare dso_local spir_func double @_Z12__spirv_LoadPi(i32 addrspace(1)*) local_unnamed_addr
declare dso_local spir_func void @_Z13__spirv_StorePiiii(i32 addrspace(1)*, i32, i32, i32) local_unnamed_addr
