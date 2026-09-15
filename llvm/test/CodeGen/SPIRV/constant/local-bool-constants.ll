; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define i1 @getConstantTrue() {
  ret i1 true
}

define i1 @getConstantFalse() {
  ret i1 false
}

; CHECK:     [[BOOL:%.+]] = OpTypeBool
; CHECK-DAG: [[FN:%.+]] = OpTypeFunction [[BOOL]]
; CHECK-DAG: [[TRUE:%.+]] = OpConstantTrue
; CHECK-DAG: [[FALSE:%.+]] = OpConstantFalse

; CHECK:     OpFunction [[BOOL]] None [[FN]]
; CHECK:     OpFunction [[BOOL]] None [[FN]]
