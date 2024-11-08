; RUN: llc -O2 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#INT16:]] = OpTypeInt 16 0
; CHECK: %[[#PARAM:]] = OpFunctionParameter %[[#INT16]]
; CHECK: OpSwitch %[[#PARAM]] %[[#]] 1 %[[#]] 2 %[[#]]

define i32 @test_switch(i16 %cond) {
entry:
  switch i16 %cond, label %default [ i16 1, label %case_one
                                     i16 2, label %case_two ]
case_one:
  ret i32 1
case_two:
  ret i32 2
default:
  ret i32 3
}
