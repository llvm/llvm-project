; RUN: llc -mtriple thumbv7-linux -relocation-model=ropi -mattr=-no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,RW-DEFAULT-MOVT,RW-DEFAULT,ROPI-MOVT,ROPI
; RUN: llc -mtriple thumbv7-linux -relocation-model=ropi -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,RW-DEFAULT-NOMOVT,RW-DEFAULT,ROPI-NOMOVT,ROPI
; RUN: llc -mtriple thumbv7-linux -relocation-model=rwpi -mattr=-no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,RO-DEFAULT-MOVT,RO-DEFAULT,RWPI-MOVT,RWPI
; RUN: llc -mtriple thumbv7-linux -relocation-model=rwpi -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,RO-DEFAULT-NOMOVT,RO-DEFAULT,RWPI-NOMOVT,RWPI
; RUN: llc -mtriple thumbv7-linux -relocation-model=ropi-rwpi -mattr=-no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ROPI-MOVT,ROPI,RWPI-MOVT,RWPI
; RUN: llc -mtriple thumbv7-linux -relocation-model=ropi-rwpi -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ROPI-NOMOVT,ROPI,RWPI-NOMOVT,RWPI

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

; RWPI-MOVT: movw r[[ADDR:[0-9]+]], :lower16:internal_global(sbrel)
; RWPI-MOVT-NEXT: movt r[[ADDR]], :upper16:internal_global(sbrel)
; RWPI-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; RWPI-NEXT: add r[[ADDR:[0-9]+]], r9
; RWPI-NEXT: ldr r0, [r[[ADDR]]]
; RWPI-NEXT: bx lr
; RWPI-NOMOVT: [[LABEL]]:
; RWPI-NOMOVT-NEXT: .long internal_global(sbrel)
entry:
  %v = load i32, ptr @internal_global
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

; RWPI-MOVT: movw r[[ADDR:[0-9]+]], :lower16:external_global(sbrel)
; RWPI-MOVT-NEXT: movt r[[ADDR]], :upper16:external_global(sbrel)
; RWPI-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; RWPI-NEXT: add r[[ADDR:[0-9]+]], r9
; RWPI-NEXT: ldr r0, [r[[ADDR]]]
; RWPI-NEXT: bx lr
; RWPI-NOMOVT: [[LABEL]]:
; RWPI-NOMOVT-NEXT: .long external_global(sbrel)
entry:
  %v = load i32, ptr @external_global
  ret i32 %v
}

@internal_constant = internal constant i32 42
define i32 @test_internal_constant() {
; CHECK-LABEL: test_internal_constant:
; ROPI-MOVT: movw r[[ADDR:[0-9]+]], :lower16:(internal_constant-([[ANCHOR:.L[[:alnum:]_]+]]+4)
; ROPI-MOVT-NEXT: movt r[[ADDR]], :upper16:(internal_constant-([[ANCHOR]]+4)
; ROPI-MOVT-NEXT: [[ANCHOR]]:
; ROPI-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ROPI-NOMOVT-NEXT: [[ANCHOR:.L[[:alnum:]_]+]]:
; ROPI-NEXT: add r[[ADDR:[0-9]+]], pc
; ROPI-NEXT: ldr r0, [r[[ADDR]]]
; ROPI-NEXT: bx lr
; ROPI-NOMOVT: [[LABEL]]:
; ROPI-NOMOVT-NEXT: .long internal_constant-([[ANCHOR]]+4)

; RO-DEFAULT-MOVT: movw r[[ADDR:[0-9]+]], :lower16:internal_constant
; RO-DEFAULT-MOVT-NEXT: movt r[[ADDR]], :upper16:internal_constant
; RO-DEFAULT-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; RO-DEFAULT-NEXT: ldr r0, [r[[ADDR]]]
; RO-DEFAULT-NEXT: bx lr
; RO-DEFAULT-NOMOVT: [[LABEL]]:
; RO-DEFAULT-NOMOVT-NEXT: .long internal_constant
entry:
  %v = load i32, ptr @internal_constant
  ret i32 %v
}

@external_constant = external constant i32
define i32 @test_external_constant() {
; CHECK-LABEL: test_external_constant:
; ROPI-MOVT: movw r[[ADDR:[0-9]+]], :lower16:(external_constant-([[ANCHOR:.L[[:alnum:]_]+]]+4)
; ROPI-MOVT-NEXT: movt r[[ADDR]], :upper16:(external_constant-([[ANCHOR]]+4)
; ROPI-MOVT-NEXT: [[ANCHOR]]:
; ROPI-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ROPI-NOMOVT-NEXT: [[ANCHOR:.L[[:alnum:]_]+]]:
; ROPI-NEXT: add r[[ADDR:[0-9]+]], pc
; ROPI-NEXT: ldr r0, [r[[ADDR]]]
; ROPI-NEXT: bx lr
; ROPI-NOMOVT: [[LABEL]]:
; ROPI-NOMOVT-NEXT: .long external_constant-([[ANCHOR]]+4)

; RO-DEFAULT-MOVT: movw r[[ADDR:[0-9]+]], :lower16:external_constant
; RO-DEFAULT-MOVT-NEXT: movt r[[ADDR]], :upper16:external_constant
; RO-DEFAULT-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; RO-DEFAULT-NEXT: ldr r0, [r[[ADDR]]]
; RO-DEFAULT-NEXT: bx lr
; RO-DEFAULT-NOMOVT: [[LABEL]]:
; RO-DEFAULT-NOMOVT: .long external_constant
entry:
  %v = load i32, ptr @external_constant
  ret i32 %v
}

; RW-DEFAULT-NOMOVT: internal_global:
; RW-DEFAULT-NOMOVT: .long 42
; RW-DEFAULT-NOMOVT: .size internal_global, 4

; RWPI-NOMOVT: internal_global:
; RWPI-NOMOVT: .long 42
; RWPI-NOMOVT: .size internal_global, 4

; ROPI-NOMOVT: internal_constant:
; ROPI-NOMOVT: .long 42
; ROPI-NOMOVT: .size internal_constant, 4

; RO-DEFAULT-NOMOVT: internal_constant:
; RO-DEFAULT-NOMOVT: .long 42
; RO-DEFAULT-NOMOVT: .size internal_constant, 4
