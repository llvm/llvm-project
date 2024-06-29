; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#zero:]] = OpConstant %[[#int_32]] 0
; CHECK-DAG: %[[#one:]] = OpConstant %[[#int_32]] 1
; CHECK-DAG: %[[#ptr:]] = OpTypePointer CrossWorkgroup %[[#float]]

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#ptr]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#cmp_res:]] = OpSGreaterThan %[[#bool]] %[[#B]] %[[#zero]]
; CHECK: %[[#select_res:]] = OpSelect %[[#int_32]] %[[#cmp_res]] %[[#one]] %[[#zero]]
; CHECK: %[[#stof_res:]] = OpConvertSToF %[[#]] %[[#select_res]]
; CHECK: OpStore %[[#A]] %[[#stof_res]]

define dso_local spir_kernel void @K(ptr addrspace(1) nocapture %A, i32 %B) local_unnamed_addr {
entry:
  %cmp = icmp sgt i32 %B, 0
  %conv = sitofp i1 %cmp to float
  store float %conv, float addrspace(1)* %A, align 4
  ret void
}
