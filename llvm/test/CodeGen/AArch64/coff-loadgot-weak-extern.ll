; RUN: llc -mtriple=aarch64-windows-gnu < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck -check-prefix=ELF %s

; A weak external may be resolved by the linker to a definition from another
; object file whose alignment cannot be relied upon (e.g. a COFF weak external
; default fallback). The GOT-style reference must therefore not use a scaled
; LDR with the page offset folded into the instruction, because that requires
; the resolved symbol to be sufficiently aligned. Use an unscaled LDR through
; an ADD-computed address instead, which has no alignment requirement.

declare extern_weak dso_local ptr @foo()

define dso_local ptr @call_weak() {
entry:
  %r = call ptr @foo()
  ret ptr %r
}

; CHECK-LABEL: call_weak:
; CHECK-NOT: ldr {{x[0-9]+}}, [{{x[0-9]+}}, :lo12:foo
; CHECK: adrp [[REG:x[0-9]+]], foo
; CHECK-NEXT: add [[REG]], [[REG]], :lo12:foo
; CHECK-NEXT: ldr [[REG]], [[[REG]]]
; CHECK-NEXT: blr [[REG]]

; On ELF an undefined weak callee can be resolved by the linker directly, so
; a plain call is emitted.
; ELF-LABEL: call_weak:
; ELF: bl foo
