; A non-intrinsic aggregate-returning call is already a single value-id
; (OpFunctionCall returns one result id), so an aggregate PHI arm fed by one
; must NOT be reassembled: no OpCompositeInsert is emitted for the arms.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I64]]

; CHECK: %[[#]] = OpFunctionCall %[[#STRUCT]]
; CHECK-NOT: OpCompositeInsert
; CHECK: %[[#PHI:]] = OpPhi %[[#STRUCT]]
; CHECK: OpCompositeExtract %[[#I64]] %[[#PHI]] 0
define i64 @phi_call_aggregate(i1 %c) {
entry:
  br i1 %c, label %bb0, label %bb1

bb0:
  %0 = call { i64 } @producer()
  br label %epilog

bb1:
  %1 = call { i64 } @producer()
  br label %epilog

epilog:
  %2 = phi { i64 } [ %0, %bb0 ], [ %1, %bb1 ]
  %3 = extractvalue { i64 } %2, 0
  ret i64 %3
}

declare { i64 } @producer()
