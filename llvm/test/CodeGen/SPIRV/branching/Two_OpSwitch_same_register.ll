; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

define spir_kernel void @test_two_switch_same_register(i32 %value) {
; CHECK-SPIRV:      OpSwitch %[[#REGISTER:]] %[[#DEFAULT1:]] 1 %[[#CASE1:]] 0 %[[#CASE2:]]
  switch i32 %value, label %default1 [
    i32 1, label %case1
    i32 0, label %case2
  ]

; CHECK-SPIRV:      %[[#CASE1]] = OpLabel
case1:
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT1]]
  br label %default1

; CHECK-SPIRV:      %[[#CASE2]] = OpLabel
case2:
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT1]]
  br label %default1

; CHECK-SPIRV:      %[[#DEFAULT1]] = OpLabel
default1:
; CHECK-SPIRV-NEXT:      OpSwitch %[[#REGISTER]] %[[#DEFAULT2:]] 0 %[[#CASE3:]] 1 %[[#CASE4:]]
  switch i32 %value, label %default2 [
    i32 0, label %case3
    i32 1, label %case4
  ]

; CHECK-SPIRV:      %[[#CASE3]] = OpLabel
case3:
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT2]]
  br label %default2

; CHECK-SPIRV:      %[[#CASE4]] = OpLabel
case4:
; CHECK-SPIRV-NEXT: OpBranch %[[#DEFAULT2]]
  br label %default2

; CHECK-SPIRV:      %[[#DEFAULT2]] = OpLabel
default2:
; CHECK-SPIRV-NEXT: OpReturn
  ret void
}
