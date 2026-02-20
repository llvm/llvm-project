; RUN: llc < %s -mcpu=mvp -mattr=-bulk-memory,atomics | FileCheck %s --check-prefixes NO-BULK-MEM
; RUN: llc < %s -mcpu=mvp -mattr=+bulk-memory,atomics | FileCheck %s --check-prefixes BULK-MEM
; RUN: llc < %s -mcpu=mvp -mattr=+component-model-thread-context,-bulk-memory,atomics | FileCheck %s --check-prefixes NO-BULK-MEM-CMTC
; RUN: llc < %s -mcpu=mvp -mattr=+component-model-thread-context,bulk-memory,atomics | FileCheck %s --check-prefixes BULK-MEM-CMTC

; Test that the target features section contains -atomics or +atomics
; for modules that have thread local storage in their source.

target triple = "wasm32-unknown-unknown"

@foo = internal thread_local global i32 0

; -bulk-memory
; NO-BULK-MEM-LABEL: .custom_section.target_features,"",@
; NO-BULK-MEM-NEXT: .int8 3
; NO-BULK-MEM-NEXT: .int8 43
; NO-BULK-MEM-NEXT: .int8 7
; NO-BULK-MEM-NEXT: .ascii "atomics"
; NO-BULK-MEM-NEXT: .int8 45
; NO-BULK-MEM-NEXT: .int8 30
; NO-BULK-MEM-NEXT: .ascii "component-model-thread-context"
; NO-BULK-MEM-NEXT: .int8 45
; NO-BULK-MEM-NEXT: .int8 10
; NO-BULK-MEM-NEXT: .ascii "shared-mem"
; NO-BULK-MEM-NEXT: .bss.foo,"",@

; +bulk-memory
; BULK-MEM-LABEL: .custom_section.target_features,"",@
; BULK-MEM-NEXT: .int8 4
; BULK-MEM-NEXT: .int8 43
; BULK-MEM-NEXT: .int8 7
; BULK-MEM-NEXT: .ascii "atomics"
; BULK-MEM-NEXT: .int8 43
; BULK-MEM-NEXT: .int8 11
; BULK-MEM-NEXT: .ascii "bulk-memory"
; BULK-MEM-NEXT: .int8 43
; BULK-MEM-NEXT: .int8 15
; BULK-MEM-NEXT: .ascii "bulk-memory-opt"
; BULK-MEM-NEXT: .int8 45
; BULK-MEM-NEXT: .int8 30
; BULK-MEM-NEXT: .ascii "component-model-thread-context"
; BULK-MEM-NEXT: .tbss.foo,"T",@

; -bulk-memory,+component-model-thread-context
; NO-BULK-MEM-CMTC-LABEL: .custom_section.target_features,"",@
; NO-BULK-MEM-CMTC-NEXT: .int8 3
; NO-BULK-MEM-CMTC-NEXT: .int8 43
; NO-BULK-MEM-CMTC-NEXT: .int8 7
; NO-BULK-MEM-CMTC-NEXT: .ascii "atomics"
; NO-BULK-MEM-CMTC-NEXT: .int8 43
; NO-BULK-MEM-CMTC-NEXT: .int8 30
; NO-BULK-MEM-CMTC-NEXT: .ascii "component-model-thread-context"
; NO-BULK-MEM-CMTC-NEXT: .int8 45
; NO-BULK-MEM-CMTC-NEXT: .int8 10
; NO-BULK-MEM-CMTC-NEXT: .ascii "shared-mem"
; NO-BULK-MEM-CMTC-NEXT: .bss.foo,"",@

; +bulk-memory,+component-model-thread-context
; BULK-MEM-CMTC-LABEL: .custom_section.target_features,"",@
; BULK-MEM-CMTC-NEXT: .int8 4
; BULK-MEM-CMTC-NEXT: .int8 43
; BULK-MEM-CMTC-NEXT: .int8 7
; BULK-MEM-CMTC-NEXT: .ascii "atomics"
; BULK-MEM-CMTC-NEXT: .int8 43
; BULK-MEM-CMTC-NEXT: .int8 11
; BULK-MEM-CMTC-NEXT: .ascii "bulk-memory"
; BULK-MEM-CMTC-NEXT: .int8 43
; BULK-MEM-CMTC-NEXT: .int8 15
; BULK-MEM-CMTC-NEXT: .ascii "bulk-memory-opt"
; BULK-MEM-CMTC-NEXT: .int8 43
; BULK-MEM-CMTC-NEXT: .int8 30
; BULK-MEM-CMTC-NEXT: .ascii "component-model-thread-context"
; BULK-MEM-CMTC-NEXT: .tbss.foo,"T",@