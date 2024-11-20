; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define i32 @test_switch_branches(i32 %a) {
entry:
  %alloc = alloca i32
; CHECK-SPIRV:      OpSwitch %[[#]] %[[#DEFAULT:]] 1 %[[#CASE1:]] 2 %[[#CASE2:]] 3 %[[#CASE3:]]
  switch i32 %a, label %default [
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
  ]

case1:
  store i32 1, ptr %alloc
  br label %end

case2:
  store i32 2, ptr %alloc
  br label %end

case3:
  store i32 3, ptr %alloc
  br label %end

default:
  store i32 0, ptr %alloc
  br label %end

end:
  %result = load i32, ptr %alloc
  ret i32 %result

; CHECK-SPIRV:      %[[#DEFAULT]] = OpLabel
; CHECK-SPIRV:      OpBranch %[[#END:]]

; CHECK-SPIRV:      %[[#CASE1]] = OpLabel
; CHECK-SPIRV:      OpBranch %[[#END]]

; CHECK-SPIRV:      %[[#CASE2]] = OpLabel
; CHECK-SPIRV:      OpBranch %[[#END]]

; CHECK-SPIRV:      %[[#CASE3]] = OpLabel
; CHECK-SPIRV:      OpBranch %[[#END]]

; CHECK-SPIRV:      %[[#END]] = OpLabel
; CHECK-SPIRV:                  OpReturnValue
}
