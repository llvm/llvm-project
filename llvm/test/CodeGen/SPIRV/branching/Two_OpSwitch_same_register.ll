; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define spir_kernel void @test_two_switch_same_register(i32 %value) {
; CHECK-SPIRV:      OpSwitch %[[#REGISTER:]] %[[#DEFAULT1:]] 1 %[[#CASE1:]] 0 %[[#CASE2:]]
  switch i32 %value, label %default1 [
    i32 1, label %case1
    i32 0, label %case2
  ]

case1:
  br label %default1

case2:
  br label %default1

default1:
  switch i32 %value, label %default2 [
    i32 0, label %case3
    i32 1, label %case4
  ]

case3:
  br label %default2

case4:
  br label %default2

default2:
  ret void

; CHECK-SPIRV:      %[[#CASE1]] = OpLabel
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT1]]

; CHECK-SPIRV:      %[[#CASE2]] = OpLabel
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT1]]

; CHECK-SPIRV:      %[[#DEFAULT1]] = OpLabel
; CHECK-SPIRV-NEXT:      OpSwitch %[[#REGISTER]] %[[#DEFAULT2:]] 0 %[[#CASE3:]] 1 %[[#CASE4:]]

; CHECK-SPIRV:      %[[#CASE3]] = OpLabel
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT2]]

; CHECK-SPIRV:      %[[#CASE4:]] = OpLabel
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT2]]

; CHECK-SPIRV:      %[[#DEFAULT2]] = OpLabel
; CHECK-SPIRV-NEXT: OpReturn
}
