; RUN: llc < %s -mcpu=mvp | FileCheck %s --check-prefixes MVP
; RUN: llc < %s -mcpu=generic | FileCheck %s --check-prefixes GENERIC
; RUN: llc < %s | FileCheck %s --check-prefixes GENERIC
; RUN: llc < %s -mcpu=bleeding-edge | FileCheck %s --check-prefixes BLEEDING-EDGE

; Test that the target features section contains the correct set of features
; depending on -mcpu= options.

target triple = "wasm32-unknown-unknown"

; mvp: should not contain the target features section
; MVP-NOT: .custom_section.target_features,"",@

; generic: +mutable-globals, +sign-ext
; GENERIC-LABEL: .custom_section.target_features,"",@
; GENERIC-NEXT: .int8  2
; GENERIC-NEXT: .int8  43
; GENERIC-NEXT: .int8  15
; GENERIC-NEXT: .ascii  "mutable-globals"
; GENERIC-NEXT: .int8  43
; GENERIC-NEXT: .int8  8
; GENERIC-NEXT: .ascii  "sign-ext"

; bleeding-edge: +atomics, +bulk-memory, +mutable-globals, +nontrapping-fptoint,
;                +sign-ext, +simd128, +tail-call
; BLEEDING-EDGE-LABEL: .section  .custom_section.target_features,"",@
; BLEEDING-EDGE-NEXT: .int8  7
; BLEEDING-EDGE-NEXT: .int8  43
; BLEEDING-EDGE-NEXT: .int8  7
; BLEEDING-EDGE-NEXT: .ascii  "atomics"
; BLEEDING-EDGE-NEXT: .int8  43
; BLEEDING-EDGE-NEXT: .int8  11
; BLEEDING-EDGE-NEXT: .ascii  "bulk-memory"
; BLEEDING-EDGE-NEXT: .int8  43
; BLEEDING-EDGE-NEXT: .int8  15
; BLEEDING-EDGE-NEXT: .ascii  "mutable-globals"
; BLEEDING-EDGE-NEXT: .int8  43
; BLEEDING-EDGE-NEXT: .int8  19
; BLEEDING-EDGE-NEXT: .ascii  "nontrapping-fptoint"
; BLEEDING-EDGE-NEXT: .int8  43
; BLEEDING-EDGE-NEXT: .int8  8
; BLEEDING-EDGE-NEXT: .ascii  "sign-ext"
; BLEEDING-EDGE-NEXT: .int8  43
; BLEEDING-EDGE-NEXT: .int8  7
; BLEEDING-EDGE-NEXT: .ascii  "simd128"
; BLEEDING-EDGE-NEXT: .int8  43
; BLEEDING-EDGE-NEXT: .int8  9
; BLEEDING-EDGE-NEXT: .ascii  "tail-call"
