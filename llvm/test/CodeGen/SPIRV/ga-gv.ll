; RUN: llc -O0 -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj < %s  | spirv-val %}

@glob = addrspace(1) global i32 0
@glob_alias = alias i32, ptr addrspace(1) @glob

define spir_kernel void @kernel() addrspace(4) {
; CHECK: OpName %9 "kernel"
; CHECK-NEXT: OpName %8 "glob"
; CHECK-NEXT: OpName %2 "entry"
; CHECK-NEXT: OpDecorate %8 LinkageAttributes "glob" Export
; CHECK-NEXT: %3 = OpTypeVoid
; CHECK-NEXT: %4 = OpTypeFunction %3
; CHECK-NEXT: %5 = OpTypeInt 32 0
; CHECK-NEXT: %6 = OpTypePointer CrossWorkgroup %5
; CHECK-NEXT: %7 = OpConstantNull %5
; CHECK-NEXT: %8 = OpVariable %6 CrossWorkgroup %7
; CHECK-NEXT: %9 = OpFunction %3 None %4              ; -- Begin function kernel
; CHECK-NEXT: %2 = OpLabel
; CHECK-NEXT: OpStore %8 %7 Aligned 4
entry:
  store i32 0, ptr addrspace(1) @glob_alias
  ret void
}
