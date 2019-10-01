; RUN: llc -mtriple thumbv7-linux -relocation-model=pic -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ELF
; RUN: llc -mtriple thumbv7-linux -relocation-model=pic -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ELF
; RUN: llc -mtriple thumbv7-darwin -relocation-model=pic -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,DARWIN,DARWIN-MOVT
; RUN: llc -mtriple thumbv7-darwin -relocation-model=pic -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,DARWIN,DARWIN-NOMOVT

@internal_global = internal global i32 42
define i32 @test_internal_global() {
; CHECK-LABEL: test_internal_global:
; ELF: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; DARWIN-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw r[[ADDR:[0-9]+]], :lower16:(_internal_global-([[ANCHOR:L[[:alnum:]_]+]]+4))
; DARWIN-MOVT-NEXT: movt r[[ADDR]], :upper16:(_internal_global-([[ANCHOR]]+4))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; CHECK-NEXT: add r[[ADDR:[0-9]+]], pc
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF-NEXT: .long internal_global-([[ANCHOR]]+4)
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT-NEXT: .long _internal_global-([[ANCHOR]]+4)
; DARWIN-MOVT-NOT: .long _internal_global

entry:
  %v = load i32, i32* @internal_global
  ret i32 %v
}

@external_global = external global i32
define i32 @test_external_global() {
; CHECK-LABEL: test_external_global:
; ELF: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; DARWIN-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw r[[ADDR:[0-9]+]], :lower16:(L_external_global$non_lazy_ptr-([[ANCHOR:L[[:alnum:]_]+]]+4))
; DARWIN-MOVT: movt r[[ADDR]], :upper16:(L_external_global$non_lazy_ptr-([[ANCHOR]]+4))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; CHECK-NEXT: add r[[ADDR:[0-9]+]], pc
; CHECK-NEXT: ldr r[[ADDR:[0-9]+]], [r[[ADDR]]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF: [[TMPLABEL:.L[[:alnum:]_]+]]:
; ELF: .long external_global(GOT_PREL)-(([[ANCHOR]]+4)-[[TMPLABEL]])
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT: .long L_external_global$non_lazy_ptr-([[ANCHOR]]+4)
; DARWIN-NOMOVT-NOT: .long L_external_global
entry:
  %v = load i32, i32* @external_global
  ret i32 %v
}

@internal_constant = internal constant i32 42
define i32 @test_internal_constant() {
; CHECK-LABEL: test_internal_constant:
; ELF: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; DARWIN-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw r[[ADDR:[0-9]+]], :lower16:(_internal_constant-([[ANCHOR:L[[:alnum:]_]+]]+4))
; DARWIN-MOVT-NEXT: movt r[[ADDR]], :upper16:(_internal_constant-([[ANCHOR]]+4))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; CHECK-NEXT: add r[[ADDR:[0-9]+]], pc
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF-NEXT: .long internal_constant-([[ANCHOR]]+4)
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT-NEXT: .long _internal_constant-([[ANCHOR]]+4)
; DARWIN-MOVT-NOT: .long _internal_constant

entry:
  %v = load i32, i32* @internal_constant
  ret i32 %v
}

@external_constant = external constant i32
define i32 @test_external_constant() {
; CHECK-LABEL: test_external_constant:
; ELF: ldr r[[ADDR:[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; DARWIN-NOMOVT: ldr r[[ADDR:[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw r[[ADDR:[0-9]+]], :lower16:(L_external_constant$non_lazy_ptr-([[ANCHOR:L[[:alnum:]_]+]]+4))
; DARWIN-MOVT: movt r[[ADDR]], :upper16:(L_external_constant$non_lazy_ptr-([[ANCHOR]]+4))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; CHECK-NEXT: add r[[ADDR:[0-9]+]], pc
; CHECK-NEXT: ldr r[[ADDR:[0-9]+]], [r[[ADDR]]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF: [[TMPLABEL:.L[[:alnum:]_]+]]:
; ELF: .long external_constant(GOT_PREL)-(([[ANCHOR]]+4)-[[TMPLABEL]])
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT: .long L_external_constant$non_lazy_ptr-([[ANCHOR]]+4)
; DARWIN-NOMOVT-NOT: .long L_external_constant
entry:
  %v = load i32, i32* @external_constant
  ret i32 %v
}

; ELF: internal_global:
; DARWIN: _internal_global:
; CHECK: .long 42
; ELF: .size internal_global, 4

; ELF: internal_constant:
; DARWIN: _internal_constant:
; CHECK: .long 42
; ELF: .size internal_constant, 4

; DARWIN-DAG: L_external_global$non_lazy_ptr:
; DARWIN-DAG: .indirect_symbol _external_global

; DARWIN-DAG: L_external_constant$non_lazy_ptr:
; DARWIN-DAG: .indirect_symbol _external_constant
