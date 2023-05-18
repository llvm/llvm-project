; RUN: llc -mtriple thumbv7-linux -relocation-model=static -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ELF,ELF-MOVT
; RUN: llc -mtriple thumbv7-linux -relocation-model=static -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ELF,ELF-NOMOVT
; RUN: llc -mtriple thumbv7-darwin -relocation-model=static -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,DARWIN,DARWIN-MOVT
; RUN: llc -mtriple thumbv7-darwin -relocation-model=static -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,DARWIN,DARWIN-NOMOVT

@internal_global = internal global i32 42
define i32 @test_internal_global() {
; CHECK-LABEL: test_internal_global:
; ELF-MOVT: movw r[[ADDR:[0-9]+]], :lower16:internal_global
; ELF-MOVT-NEXT: movt r[[ADDR]], :upper16:internal_global
; ELF-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; DARWIN-MOVT: movw r[[ADDR:[0-9]+]], :lower16:_internal_global
; DARWIN-MOVT-NEXT: movt r[[ADDR]], :upper16:_internal_global
; DARWIN-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF-NOMOVT: [[LABEL]]:
; ELF-NOMOVT-NEXT: .long internal_global
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT-NEXT: .long _internal_global

entry:
  %v = load i32, ptr @internal_global
  ret i32 %v
}

@external_global = external global i32
define i32 @test_external_global() {
; CHECK-LABEL: test_external_global:
; ELF-MOVT: movw r[[ADDR:[0-9]+]], :lower16:external_global
; ELF-MOVT-NEXT: movt r[[ADDR]], :upper16:external_global
; ELF-NOMOVT: ldr r[[ADDR:[0-9]+]], [[CONST_POOL:.L[[:alnum:]_]+]]
; DARWIN-MOVT: movw r[[ADDR:[0-9]+]], :lower16:_external_global
; DARWIN-MOVT: movt r[[ADDR]], :upper16:_external_global
; DARWIN-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF-NOMOVT: [[CONST_POOL]]:
; ELF-NOMOVT: .long external_global
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT: .long _external_global
entry:
  %v = load i32, ptr @external_global
  ret i32 %v
}

; ELF: internal_global:
; DARWIN: _internal_global:
; CHECK: .long 42
; ELF: .size internal_global, 4
