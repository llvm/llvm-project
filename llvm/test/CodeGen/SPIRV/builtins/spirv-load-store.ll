; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
;; Translate SPIR-V friendly OpLoad and OpStore calls

; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#INT32]] 42
; CHECK-DAG: %[[#PTRINT32:]] = OpTypePointer CrossWorkgroup %[[#INT32]]
; CHECK: %[[#BITCASTorPARAMETER:]] = {{OpBitcast|OpFunctionParameter}}{{.*}}%[[#PTRINT32]]{{.*}}

; CHECK: OpStore %[[#BITCASTorPARAMETER]] %[[#CONST]] Volatile|Aligned 4
; CHECK: %[[#]] = OpLoad %[[#]] %[[#BITCASTorPARAMETER]]

define weak_odr dso_local spir_kernel void @foo(i32 addrspace(1)* %var) {
entry:
  tail call spir_func void @_Z13__spirv_StorePiiii(i32 addrspace(1)* %var, i32 42, i32 3, i32 4)
  %value = tail call spir_func double @_Z12__spirv_LoadPi(i32 addrspace(1)* %var)
  ret void
}

declare dso_local spir_func double @_Z12__spirv_LoadPi(i32 addrspace(1)*) local_unnamed_addr
declare dso_local spir_func void @_Z13__spirv_StorePiiii(i32 addrspace(1)*, i32, i32, i32) local_unnamed_addr
