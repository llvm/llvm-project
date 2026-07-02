; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-unknown-eabi < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s

define i32 @cr0_input() {
; CHECK-LABEL: cr0_input:
; CHECK: {{mto?crf}} 128, {{[0-9]+}}
; CHECK: #APP
; CHECK-NEXT: rfi
; CHECK-NEXT: #NO_APP
entry:
  %0 = call i32 asm sideeffect "rfi", "={r3},{r3},{cr0}"(i32 0, i1 false)
  ret i32 %0
}

define i32 @cr1_input() {
; CHECK-LABEL: cr1_input:
; CHECK: li [[REG:[0-9]+]], 1
; CHECK: rotlwi [[REG]], [[REG]], 24
; CHECK: {{mto?crf}} 64, [[REG]]
; CHECK: #APP
; CHECK-NEXT: rfi
; CHECK-NEXT: #NO_APP
entry:
  %0 = call i32 asm sideeffect "rfi", "={r3},{r3},{cr1}"(i32 0, i1 true)
  ret i32 %0
}

define i32 @cr7_input() {
; CHECK-LABEL: cr7_input:
; CHECK: li [[REG:[0-9]+]], 1
; CHECK: {{mto?crf}} 1, [[REG]]
; CHECK: #APP
; CHECK-NEXT: rfi
; CHECK-NEXT: #NO_APP
entry:
  %0 = call i32 asm sideeffect "rfi", "={r3},{r3},{cr7}"(i32 0, i1 true)
  ret i32 %0
}
