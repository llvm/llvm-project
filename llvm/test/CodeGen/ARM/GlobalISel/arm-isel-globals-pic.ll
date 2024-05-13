; RUN: llc -mtriple armv7-linux -relocation-model=pic -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ELF
; RUN: llc -mtriple armv7-linux -relocation-model=pic -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,ELF
; RUN: llc -mtriple armv7-darwin -relocation-model=pic -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,DARWIN,DARWIN-MOVT
; RUN: llc -mtriple armv7-darwin -relocation-model=pic -mattr=+no-movt -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,DARWIN,DARWIN-NOMOVT

@internal_global = internal global i32 42
define i32 @test_internal_global() {
; CHECK-LABEL: test_internal_global:
; ELF: ldr [[OFFSET:r[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; DARWIN-NOMOVT: ldr [[OFFSET:r[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw [[OFFSET:r[0-9]+]], :lower16:(_internal_global-([[ANCHOR:L[[:alnum:]_]+]]+8))
; DARWIN-MOVT-NEXT: movt [[OFFSET]], :upper16:(_internal_global-([[ANCHOR]]+8))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; CHECK-NEXT: add r[[ADDR:[0-9]+]], pc, [[OFFSET]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF-NEXT: .long internal_global-([[ANCHOR]]+8)
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT-NEXT: .long _internal_global-([[ANCHOR]]+8)
; DARWIN-MOVT-NOT: .long _internal_global

entry:
  %v = load i32, ptr @internal_global
  ret i32 %v
}

@external_global = external global i32
define i32 @test_external_global() {
; CHECK-LABEL: test_external_global:
; ELF: ldr [[OFFSET:r[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; ELF-NEXT: ldr r[[ADDR:[0-9]+]], [pc, [[OFFSET]]]
; DARWIN-NOMOVT: ldr [[OFFSET:r[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw [[OFFSET:r[0-9]+]], :lower16:(L_external_global$non_lazy_ptr-([[ANCHOR:L[[:alnum:]_]+]]+8))
; DARWIN-MOVT: movt [[OFFSET]], :upper16:(L_external_global$non_lazy_ptr-([[ANCHOR]]+8))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; DARWIN: ldr r[[ADDR:[0-9]+]], [pc, [[OFFSET]]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF: [[TMPLABEL:.L[[:alnum:]_]+]]:
; ELF: .long external_global(GOT_PREL)-(([[ANCHOR]]+8)-[[TMPLABEL]])
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT: .long L_external_global$non_lazy_ptr-([[ANCHOR]]+8)
; DARWIN-NOMOVT-NOT: .long L_external_global
entry:
  %v = load i32, ptr @external_global
  ret i32 %v
}

@internal_constant = internal constant i32 42
define i32 @test_internal_constant() {
; CHECK-LABEL: test_internal_constant:
; ELF: ldr [[OFFSET:r[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; DARWIN-NOMOVT: ldr [[OFFSET:r[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw [[OFFSET:r[0-9]+]], :lower16:(_internal_constant-([[ANCHOR:L[[:alnum:]_]+]]+8))
; DARWIN-MOVT-NEXT: movt [[OFFSET]], :upper16:(_internal_constant-([[ANCHOR]]+8))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; CHECK-NEXT: add r[[ADDR:[0-9]+]], pc, [[OFFSET]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF-NEXT: .long internal_constant-([[ANCHOR]]+8)
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT-NEXT: .long _internal_constant-([[ANCHOR]]+8)
; DARWIN-MOVT-NOT: .long _internal_constant

entry:
  %v = load i32, ptr @internal_constant
  ret i32 %v
}

@external_constant = external constant i32
define i32 @test_external_constant() {
; CHECK-LABEL: test_external_constant:
; ELF: ldr [[OFFSET:r[0-9]+]], [[LABEL:.L[[:alnum:]_]+]]
; ELF: [[ANCHOR:.L[[:alnum:]_]+]]:
; ELF-NEXT: ldr r[[ADDR:[0-9]+]], [pc, [[OFFSET]]]
; DARWIN-NOMOVT: ldr [[OFFSET:r[0-9]+]], [[LABEL:L[[:alnum:]_]+]]
; DARWIN-MOVT: movw [[OFFSET:r[0-9]+]], :lower16:(L_external_constant$non_lazy_ptr-([[ANCHOR:L[[:alnum:]_]+]]+8))
; DARWIN-MOVT: movt [[OFFSET]], :upper16:(L_external_constant$non_lazy_ptr-([[ANCHOR]]+8))
; DARWIN: [[ANCHOR:L[[:alnum:]_]+]]:
; DARWIN: ldr r[[ADDR:[0-9]+]], [pc, [[OFFSET]]]
; CHECK-NEXT: ldr r0, [r[[ADDR]]]
; CHECK-NEXT: bx lr
; ELF: [[LABEL]]:
; ELF: [[TMPLABEL:.L[[:alnum:]_]+]]:
; ELF: .long external_constant(GOT_PREL)-(([[ANCHOR]]+8)-[[TMPLABEL]])
; DARWIN-NOMOVT: [[LABEL]]:
; DARWIN-NOMOVT: .long L_external_constant$non_lazy_ptr-([[ANCHOR]]+8)
; DARWIN-NOMOVT-NOT: .long L_external_constant
entry:
  %v = load i32, ptr @external_constant
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
