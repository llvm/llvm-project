; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#pointer:]] = OpTypePointer CrossWorkgroup %[[#float]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#]]

define void @foo(float addrspace(1)* %A, i32 %B) {
  %cmp = icmp sgt i32 %B, 0
  %conv = uitofp i1 %cmp to float
; CHECK: %[[#utof_res:]] = OpConvertUToF %[[#float]] %[[#]]
; CHECK: %[[#bitcast:]] = OpBitcast %[[#pointer]] %[[#A]]
; CHECK: OpStore %[[#bitcast]] %[[#utof_res]]
  %BC1 = bitcast float addrspace(1)* %A to i32 addrspace(1)*
  %BC2 = bitcast i32 addrspace(1)* %BC1 to float addrspace(1)*
  store float %conv, float addrspace(1)* %BC2, align 4;
  ret void
}
