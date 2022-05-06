; RUN: llc -O0 %s -o - | FileCheck %s

; NOTE: This does not check for structured control-flow operations.

target triple = "spirv32-unknown-unknown"

; CHECK-DAG: OpName [[FOO:%.+]] "foo"
; CHECK-DAG: OpName [[BAR:%.+]] "bar"

; CHECK-DAG: [[I32:%.+]] = OpTypeInt 32
; CHECK-DAG: [[BOOL:%.+]] = OpTypeBool

declare i32 @foo()
declare i32 @bar()

define i32 @test_if(i32 %a, i32 %b) {
entry:
  %cond = icmp eq i32 %a, %b
  br i1 %cond, label %true_label, label %false_label

true_label:
  %v1 = call i32 @foo()
  br label %merge_label

false_label:
  %v2 = call i32 @bar()
  br label %merge_label

merge_label:
  %v = phi i32 [%v1, %true_label], [%v2, %false_label]
  ret i32 %v
}

; CHECK: OpFunction
; CHECK: [[A:%.+]] = OpFunctionParameter [[I32]]
; CHECK: [[B:%.+]] = OpFunctionParameter [[I32]]

; CHECK: [[ENTRY:%.+]] = OpLabel
; CHECK: [[COND:%.+]] = OpIEqual [[BOOL]] [[A]] [[B]]
; CHECK: OpBranchConditional [[COND]] [[TRUE_LABEL:%.+]] [[FALSE_LABEL:%.+]]

; CHECK: [[TRUE_LABEL]] = OpLabel
; CHECK: [[V1:%.+]] = OpFunctionCall [[I32]] [[FOO]]
; CHECK: OpBranch [[MERGE_LABEL:%.+]]

; CHECK: [[FALSE_LABEL]] = OpLabel
; CHECK: [[V2:%.+]] = OpFunctionCall [[I32]] [[BAR]]
; CHECK: OpBranch [[MERGE_LABEL]]

; CHECK: [[MERGE_LABEL]] = OpLabel
; CHECK-NEXT: [[V:%.+]] = OpPhi [[I32]] [[V1]] [[TRUE_LABEL]] [[V2]] [[FALSE_LABEL]]
; CHECK: OpReturnValue [[V]]
; CHECK-NEXT: OpFunctionEnd
