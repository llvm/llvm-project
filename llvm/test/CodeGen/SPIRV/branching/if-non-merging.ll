; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: [[I32:%.+]] = OpTypeInt 32
; CHECK-DAG: [[BOOL:%.+]] = OpTypeBool
; CHECK-DAG: [[TRUE:%.+]] = OpConstantTrue
; CHECK-DAG: [[FALSE:%.+]] = OpConstantFalse

define i1 @test_if(i32 %a, i32 %b) {
entry:
  %cond = icmp eq i32 %a, %b
  br i1 %cond, label %true_label, label %false_label
true_label:
  ret i1 true
false_label:
  ret i1 false
}

; CHECK: OpFunction
; CHECK: [[A:%.+]] = OpFunctionParameter [[I32]]
; CHECK: [[B:%.+]] = OpFunctionParameter [[I32]]
; CHECK: [[ENTRY:%.+]] = OpLabel
; CHECK: [[COND:%.+]] = OpIEqual [[BOOL]] [[A]] [[B]]
; CHECK: OpBranchConditional [[COND]] [[TRUE_LABEL:%.+]] [[FALSE_LABEL:%.+]]

; CHECK: [[FALSE_LABEL]] = OpLabel
; CHECK: OpReturnValue [[FALSE]]

; CHECK: [[TRUE_LABEL]] = OpLabel
; CHECK: OpReturnValue [[TRUE]]
