; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define void @test_switch_with_unreachable_block(i1 %a) {
  %value = zext i1 %a to i32
; CHECK-SPIRV:      OpSwitch %[[#]] %[[#UNREACHABLE:]] 0 %[[#REACHABLE:]] 1 %[[#REACHABLE:]]
  switch i32 %value, label %unreachable [
    i32 0, label %reachable
    i32 1, label %reachable
  ]

; CHECK-SPIRV:      %[[#UNREACHABLE]] = OpLabel
; CHECK-SPIRV-NEXT: OpUnreachable

; CHECK-SPIRV-NEXT: %[[#REACHABLE]] = OpLabel
reachable:
; CHECK-SPIRV-NEXT: OpReturn
  ret void

unreachable:
  unreachable
}
