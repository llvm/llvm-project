; Test that in cooperative threading mode (wasm32-wasip3), thread-local variables
; are NOT stripped even when atomics are absent.  In non-cooperative mode
; (wasm32-unknown-unknown) TLS is stripped to .bss when atomics are absent.

; RUN: llc < %s -mtriple=wasm32-wasip3 -mcpu=mvp -mattr=-atomics,+bulk-memory \
; RUN:   | FileCheck %s --check-prefixes=COOP
; RUN: llc < %s -mtriple=wasm32-unknown-unknown -mcpu=mvp -mattr=-atomics,+bulk-memory \
; RUN:   | FileCheck %s --check-prefixes=PLAIN

target triple = "wasm32-unknown-unknown"

@foo = internal thread_local global i32 0

; Cooperative threading: TLS is preserved — the section stays .tbss.
; COOP:     .tbss.foo
; COOP-NOT: .bss.foo

; Non-cooperative: TLS stripped
; PLAIN:     .bss.foo
; PLAIN-NOT: .tbss.foo
