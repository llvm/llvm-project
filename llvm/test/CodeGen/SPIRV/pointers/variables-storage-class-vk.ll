; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#U32:]] = OpTypeInt 32 0

; CHECK-DAG: %[[#VAL:]] = OpConstant %[[#U32]] 456
; CHECK-DAG: %[[#VTYPE:]] = OpTypePointer Private %[[#U32]]
; CHECK-DAG: %[[#VAR:]] = OpVariable %[[#VTYPE]] Private %[[#VAL]]
; CHECK-NOT: OpDecorate %[[#VAR]] LinkageAttributes
@PrivInternal = internal addrspace(10) global i32 456

define void @main() {
  %l = load i32, ptr addrspace(10) @PrivInternal
  ret void
}
