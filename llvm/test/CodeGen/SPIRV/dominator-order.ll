; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; This test checks that basic blocks are reordered in SPIR-V so that dominators
; are emitted ahead of their dominated blocks as required by the SPIR-V
; specification.

; CHECK-DAG: OpName %[[#ENTRY:]] "entry"
; CHECK-DAG: OpName %[[#FOR_BODY137_LR_PH:]] "for.body137.lr.ph"
; CHECK-DAG: OpName %[[#FOR_BODY:]] "for.body"
; CHECK-DAG: OpName %[[#OUTER:]] "outer"
; CHECK-DAG: OpName %[[#INNER_PREHEADER:]] "inner.preheader"
; CHECK-DAG: OpName %[[#INNER:]] "inner"
; CHECK-DAG: OpName %[[#OUTER_LOOPEXIT:]] "outer.loopexit"

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

; Check that blocks inserted by LoopSimplify (such as outer.loopexit, which is
; inserted before outer but dominated by inner) are re-sorted after their
; dominators.
; CHECK: OpFunction
; CHECK: %[[#OUTER]] = OpLabel
; CHECK: %[[#INNER_PREHEADER]] = OpLabel
; CHECK: %[[#INNER]] = OpLabel
; CHECK: %[[#OUTER_LOOPEXIT]] = OpLabel
; CHECK: OpFunctionEnd

define spir_kernel void @test_loop_simplify(i1 %c1, i1 %c2) {
entry:
  br label %outer

outer:
  br i1 %c1, label %inner, label %exit

inner:
  br i1 %c2, label %inner, label %outer

exit:
  ret void
}
