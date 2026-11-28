; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A poison aggregate PHI arm materializes in its incoming block,
; where it dominates the edge, once per edge.
; CHECK-DAG: [[TY:%.+]] = OpTypeStruct {{%.+}} {{%.+}}
; CHECK-DAG: [[UNDEF:%.+]] = OpUndef [[TY]]
; CHECK: OpFunction
; CHECK: [[ENTRY:%.+]] = OpLabel
; CHECK: OpPhi [[TY]] [[UNDEF]] [[ENTRY]]
define { i32, float } @f(i1 %c) {
entry:
  br label %next
next:
  %p = phi { i32, float } [ poison, %entry ]
  ret { i32, float } %p
}

; CHECK: OpFunction
; CHECK: [[ENTRY2:%.+]] = OpLabel
; CHECK: OpSwitch
; CHECK: [[OTHER:%.+]] = OpLabel
; CHECK: OpPhi [[TY]] [[UNDEF]] [[ENTRY2]] {{%.+}} [[OTHER]]
define { i32, float } @g(i32 %s, { i32, float } %a) {
entry:
  switch i32 %s, label %other [ i32 0, label %next
                                i32 1, label %next ]
other:
  br label %next
next:
  %p = phi { i32, float } [ poison, %entry ], [ poison, %entry ], [ %a, %other ]
  ret { i32, float } %p
}
