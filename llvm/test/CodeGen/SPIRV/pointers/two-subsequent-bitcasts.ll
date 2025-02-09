; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#pointer:]] = OpTypePointer CrossWorkgroup %[[#float]]

define void @foo(float addrspace(1)* %A, i32 %B) {
  %cmp = icmp sgt i32 %B, 0
  %conv = uitofp i1 %cmp to float
; CHECK-DAG: %[[#utof_res:]] = OpConvertUToF %[[#float]] %[[#]]
; CHECK-DAG: %[[#bitcastORparam:]] = {{OpBitcast|OpFunctionParameter}}{{.*}}%[[#pointer]]{{.*}}
; CHECK: OpStore %[[#bitcastORparam]] %[[#utof_res]]
  %BC1 = bitcast float addrspace(1)* %A to i32 addrspace(1)*
  %BC2 = bitcast i32 addrspace(1)* %BC1 to float addrspace(1)*
  store float %conv, float addrspace(1)* %BC2, align 4;
  ret void
}
