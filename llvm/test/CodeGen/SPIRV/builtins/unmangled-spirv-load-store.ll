; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#INT32]] 42
; CHECK-DAG: %[[#PTRINT8:]] = OpTypePointer CrossWorkgroup %[[#INT8]]
; CHECK: %[[#BITCASTorPARAMETER:]] = {{OpBitcast|OpFunctionParameter}}{{.*}}%[[#PTRINT8]]{{.*}}

; CHECK: OpStore %[[#BITCASTorPARAMETER]] %[[#CONST]] Volatile|Aligned 4
; CHECK: %[[#]] = OpLoad %[[#]] %[[#BITCASTorPARAMETER]]

define weak_odr dso_local spir_kernel void @foo(ptr addrspace(1) %var) {
entry:
  tail call spir_func void @__spirv_Store(ptr addrspace(1) %var, i32 42, i32 3, i32 4)
  %value = tail call spir_func double @__spirv_Load(ptr addrspace(1) %var)
  ret void
}

declare dso_local spir_func double @__spirv_Load(ptr addrspace(1)) local_unnamed_addr
declare dso_local spir_func void @__spirv_Store(ptr addrspace(1), i32, i32, i32) local_unnamed_addr
