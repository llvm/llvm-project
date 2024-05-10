; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[#GV:]] "v"
; CHECK-DAG: OpDecorate %[[#GV]] LinkageAttributes "v" Export
; CHECK-DAG: OpDecorate %[[#GV]] Constant

@v = addrspace(1) global i32 0, !spirv.Decorations !0

define spir_kernel void @foo() {
entry:
  %pv = load ptr addrspace(1), ptr addrspace(1) @v
  store i32 3, ptr addrspace(1) %pv
  ret void
}

!0 = !{!1, !2}
!1 = !{i32 22}
!2 = !{i32 41, !"v", i32 0}
