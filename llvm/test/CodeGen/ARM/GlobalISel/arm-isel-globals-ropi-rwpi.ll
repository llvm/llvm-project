; RUN: llc -mtriple armv7-linux -relocation-model=ropi -mattr=-no-movt,+v8m -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,RW-DEFAULT-MOVT,RW-DEFAULT,ROPI-MOVT,ROPI
; RUN: llc -mtriple armv7-linux -relocation-model=ropi -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,RW-DEFAULT-NOMOVT,RW-DEFAULT,ROPI-NOMOVT,ROPI

@internal_global = internal global i32 42
define i32 @test_internal_global() {
; CHECK-LABEL: test_internal_global:
; RW-DEFAULT-MOVT: movw r[[ADDR:[0-9]+]], :lower16:internal_global
; RW-DEFAULT-MOVT-NEXT: movt r[[ADDR]], :upper16:internal_global
; RW-DEFAULT-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; RW-DEFAULT-NEXT: ldr r0, [r[[ADDR]]]
; RW-DEFAULT-NEXT: bx lr
; RW-DEFAULT-NOMOVT: [[LABEL]]:
; RW-DEFAULT-NOMOVT-NEXT: .long internal_global
entry:
  %v = load i32, i32* @internal_global
  ret i32 %v
}

@external_global = external global i32
define i32 @test_external_global() {
; CHECK-LABEL: test_external_global:
; RW-DEFAULT-MOVT: movw r[[ADDR:[0-9]+]], :lower16:external_global
; RW-DEFAULT-MOVT-NEXT: movt r[[ADDR]], :upper16:external_global
; RW-DEFAULT-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; RW-DEFAULT-NEXT: ldr r0, [r[[ADDR]]]
; RW-DEFAULT-NEXT: bx lr
; RW-DEFAULT-NOMOVT: [[LABEL]]:
; RW-DEFAULT-NOMOVT: .long external_global
entry:
  %v = load i32, i32* @external_global
  ret i32 %v
}

@internal_constant = internal constant i32 42
define i32 @test_internal_constant() {
; CHECK-LABEL: test_internal_constant:
; ROPI-MOVT: movw [[OFFSET:r[0-9]+]], :lower16:(internal_constant-([[ANCHOR:.L[[:alnum:]_]+]]+8)
; ROPI-MOVT-NEXT: movt [[OFFSET]], :upper16:(internal_constant-([[ANCHOR]]+8)
; ROPI-MOVT-NEXT: [[ANCHOR]]:
; ROPI-NOMOVT: ldr [[OFFSET:r[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ROPI-NOMOVT-NEXT: [[ANCHOR:.L[[:alnum:]_]+]]:
; ROPI-NEXT: add r[[ADDR:[0-9]+]], pc, [[OFFSET]]
; ROPI-NEXT: ldr r0, [r[[ADDR]]]
; ROPI-NEXT: bx lr
; ROPI-NOMOVT: [[LABEL]]:
; ROPI-NOMOVT-NEXT: .long internal_constant-([[ANCHOR]]+8)
entry:
  %v = load i32, i32* @internal_constant
  ret i32 %v
}

@external_constant = external constant i32
define i32 @test_external_constant() {
; CHECK-LABEL: test_external_constant:
; ROPI-MOVT: movw [[OFFSET:r[0-9]+]], :lower16:(external_constant-([[ANCHOR:.L[[:alnum:]_]+]]+8)
; ROPI-MOVT-NEXT: movt [[OFFSET]], :upper16:(external_constant-([[ANCHOR]]+8)
; ROPI-MOVT-NEXT: [[ANCHOR]]:
; ROPI-NOMOVT: ldr [[OFFSET:r[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ROPI-NOMOVT-NEXT: [[ANCHOR:.L[[:alnum:]_]+]]:
; ROPI-NEXT: add r[[ADDR:[0-9]+]], pc, [[OFFSET]]
; ROPI-NEXT: ldr r0, [r[[ADDR]]]
; ROPI-NEXT: bx lr
; ROPI-NOMOVT: [[LABEL]]:
; ROPI-NOMOVT-NEXT: .long external_constant-([[ANCHOR]]+8)
entry:
  %v = load i32, i32* @external_constant
  ret i32 %v
}

; RW-DEFAULT-NOMOVT: internal_global:
; RW-DEFAULT-NOMOVT: .long 42
; RW-DEFAULT-NOMOVT: .size internal_global, 4

; ROPI-NOMOVT: internal_constant:
; ROPI-NOMOVT: .long 42
; ROPI-NOMOVT: .size internal_constant, 4
