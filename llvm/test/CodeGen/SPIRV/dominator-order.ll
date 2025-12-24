; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; This test checks that basic blocks are reordered in SPIR-V so that dominators
; are emitted ahead of their dominated blocks as required by the SPIR-V
; specification.

; CHECK-DAG: OpName %[[#ENTRY:]] "entry"
; CHECK-DAG: OpName %[[#FOR_BODY137_LR_PH:]] "for.body137.lr.ph"
; CHECK-DAG: OpName %[[#FOR_BODY:]] "for.body"

; CHECK: %[[#ENTRY]] = OpLabel
; CHECK: %[[#FOR_BODY]] = OpLabel
; CHECK: %[[#FOR_BODY137_LR_PH]] = OpLabel

define spir_kernel void @test(ptr addrspace(1) %arg, i1 %cond) {
entry:
  br label %for.body

for.body137.lr.ph:                                ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  br i1 %cond, label %for.body, label %for.body137.lr.ph
}
