; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK: %[[#BOOL:]] = OpLogicalOr %[[#BOOL]] %[[#]] %[[#]]

define spir_kernel void @foo(
  ptr addrspace(1) nocapture noundef writeonly %Dst,
  i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %a1 = trunc i32 %a to i1
  %b1 = trunc i32 %b to i1
  %ab1 = or i1 %a1, %b1
  %ab32 = zext i1 %ab1 to i32
  store i32 %ab32, ptr addrspace(1) %Dst
  ret void
}
