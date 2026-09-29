; RUN: llc < %s -mcpu=mvp -mattr=-bulk-memory,atomics | FileCheck %s --check-prefixes NO-BULK-MEM
; RUN: llc < %s -mcpu=mvp -mattr=+bulk-memory,atomics | FileCheck %s --check-prefixes BULK-MEM

; Test that the target features section contains -atomics or +atomics
; for modules that have thread local storage in their source.

target triple = "wasm32-unknown-unknown"

@foo = internal thread_local global i32 0

; -bulk-memory
; NO-BULK-MEM-LABEL: .custom_section.target_features,"",@
; NO-BULK-MEM-NEXT: .int8 2
; NO-BULK-MEM-NEXT: .int8 43
; NO-BULK-MEM-NEXT: .int8 7
; NO-BULK-MEM-NEXT: .ascii "atomics"
; NO-BULK-MEM-NEXT: .int8 45
; NO-BULK-MEM-NEXT: .int8 10
; NO-BULK-MEM-NEXT: .ascii "shared-mem"
; NO-BULK-MEM-NEXT: .bss.foo,"",@

; +bulk-memory
; BULK-MEM-LABEL: .custom_section.target_features,"",@
; BULK-MEM-NEXT: .int8 3
; BULK-MEM-NEXT: .int8 43
; BULK-MEM-NEXT: .int8 7
; BULK-MEM-NEXT: .ascii "atomics"
; BULK-MEM-NEXT: .int8 43
; BULK-MEM-NEXT: .int8 11
; BULK-MEM-NEXT: .ascii "bulk-memory"
; BULK-MEM-NEXT: .int8 43
; BULK-MEM-NEXT: .int8 15
; BULK-MEM-NEXT: .ascii "bulk-memory-opt"
; BULK-MEM-NEXT: .tbss.foo,"T",@
