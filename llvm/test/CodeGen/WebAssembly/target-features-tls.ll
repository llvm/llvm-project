; RUN: llc < %s -mattr=-atomics | FileCheck %s --check-prefixes CHECK,NO-ATOMICS
; RUN: llc < %s -mattr=+atomics | FileCheck %s --check-prefixes CHECK,ATOMICS

; Test that the target features section contains -atomics or +atomics
; for modules that have thread local storage in their source.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@foo = internal thread_local global i32 0

; CHECK-LABEL: .custom_section.target_features,"",@

; -atomics
; NO-ATOMICS-NEXT: .int8 1
; NO-ATOMICS-NEXT: .int8 45
; NO-ATOMICS-NEXT: .int8 7
; NO-ATOMICS-NEXT: .ascii "atomics"
; NO-ATOMICS-NEXT: .bss.foo,"",@

; +atomics
; ATOMICS-NEXT: .int8 1
; ATOMICS-NEXT: .int8 43
; ATOMICS-NEXT: .int8 7
; ATOMICS-NEXT: .ascii "atomics"
; ATOMICS-NEXT: .tbss.foo,"",@
