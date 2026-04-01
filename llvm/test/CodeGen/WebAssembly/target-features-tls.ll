; RUN: llc < %s -mcpu=mvp -mattr=-bulk-memory,atomics | FileCheck %s --check-prefixes NO-BULK-MEM
; RUN: llc < %s -mcpu=mvp -mattr=+bulk-memory,atomics | FileCheck %s --check-prefixes BULK-MEM
; RUN: llc < %s -mcpu=mvp -mattr=+libcall-thread-context,-bulk-memory,atomics | FileCheck %s --check-prefixes NO-BULK-MEM-LIBCALL
; RUN: llc < %s -mcpu=mvp -mattr=+libcall-thread-context,bulk-memory,atomics | FileCheck %s --check-prefixes BULK-MEM-LIBCALL

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

; -bulk-memory,+libcall-thread-context
; NO-BULK-MEM-LIBCALL-LABEL: .custom_section.target_features,"",@
; NO-BULK-MEM-LIBCALL-NEXT: .int8 3
; NO-BULK-MEM-LIBCALL-NEXT: .int8 43
; NO-BULK-MEM-LIBCALL-NEXT: .int8 7
; NO-BULK-MEM-LIBCALL-NEXT: .ascii "atomics"
; NO-BULK-MEM-LIBCALL-NEXT: .int8 43
; NO-BULK-MEM-LIBCALL-NEXT: .int8 22
; NO-BULK-MEM-LIBCALL-NEXT: .ascii "libcall-thread-context"
; NO-BULK-MEM-LIBCALL-NEXT: .int8 45
; NO-BULK-MEM-LIBCALL-NEXT: .int8 10
; NO-BULK-MEM-LIBCALL-NEXT: .ascii "shared-mem"
; NO-BULK-MEM-LIBCALL-NEXT: .bss.foo,"",@

; +bulk-memory,+libcall-thread-context
; BULK-MEM-LIBCALL-LABEL: .custom_section.target_features,"",@
; BULK-MEM-LIBCALL-NEXT: .int8 4
; BULK-MEM-LIBCALL-NEXT: .int8 43
; BULK-MEM-LIBCALL-NEXT: .int8 7
; BULK-MEM-LIBCALL-NEXT: .ascii "atomics"
; BULK-MEM-LIBCALL-NEXT: .int8 43
; BULK-MEM-LIBCALL-NEXT: .int8 11
; BULK-MEM-LIBCALL-NEXT: .ascii "bulk-memory"
; BULK-MEM-LIBCALL-NEXT: .int8 43
; BULK-MEM-LIBCALL-NEXT: .int8 15
; BULK-MEM-LIBCALL-NEXT: .ascii "bulk-memory-opt"
; BULK-MEM-LIBCALL-NEXT: .int8 43
; BULK-MEM-LIBCALL-NEXT: .int8 22
; BULK-MEM-LIBCALL-NEXT: .ascii "libcall-thread-context"
; BULK-MEM-LIBCALL-NEXT: .tbss.foo,"T",@
