; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

define i32 @test_switch_branches(i32 %a) {
entry:
  %alloc = alloca i32
; CHECK-SPIRV:      OpSwitch %[[#]] %[[#DEFAULT:]] 1 %[[#CASE1:]] 2 %[[#CASE2:]] 3 %[[#CASE3:]]
  switch i32 %a, label %default [
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
  ]

; CHECK-SPIRV:      %[[#CASE1]] = OpLabel
case1:
  store i32 1, ptr %alloc
; CHECK-SPIRV:      OpBranch %[[#END:]]
  br label %end

; CHECK-SPIRV:      %[[#CASE2]] = OpLabel
case2:
  store i32 2, ptr %alloc
; CHECK-SPIRV:      OpBranch %[[#END]]
  br label %end

; CHECK-SPIRV:      %[[#CASE3]] = OpLabel
case3:
  store i32 3, ptr %alloc
; CHECK-SPIRV:      OpBranch %[[#END]]
  br label %end

; CHECK-SPIRV:      %[[#DEFAULT]] = OpLabel
default:
  store i32 0, ptr %alloc
; CHECK-SPIRV:      OpBranch %[[#END]]
  br label %end

; CHECK-SPIRV:      %[[#END]] = OpLabel
end:
  %result = load i32, ptr %alloc
  ret i32 %result
}
