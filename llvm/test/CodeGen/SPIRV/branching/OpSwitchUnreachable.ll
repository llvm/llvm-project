; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

define void @test_switch_with_unreachable_block(i1 %a) {
  %value = zext i1 %a to i32
; CHECK-SPIRV:      OpSwitch %[[#]] %[[#REACHABLE:]]
  switch i32 %value, label %unreachable [
    i32 0, label %reachable
    i32 1, label %reachable
  ]

; CHECK-SPIRV-NEXT: %[[#REACHABLE]] = OpLabel
reachable:
; CHECK-SPIRV-NEXT: OpReturn
  ret void

; CHECK-SPIRV:      %[[#]] = OpLabel
; CHECK-SPIRV-NEXT: OpUnreachable
unreachable:
  unreachable
}
