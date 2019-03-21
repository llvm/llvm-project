; RUN: llc < %s | FileCheck %s --check-prefixes CHECK,ATTRS
; RUN: llc < %s -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN; llc < %s -mattr=+atomics | FileCheck %s --check-prefixes CHECK,ATOMICS
; RUN: llc < %s -mcpu=bleeding-edge | FileCheck %s --check-prefixes CHECK,BLEEDING-EDGE

; Test that codegen emits target features from the command line or
; function attributes correctly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @foo() #0 {
  ret void
}

define void @bar() #1 {
  ret void
}

attributes #0 = { "target-features"="+sign-ext" }
attributes #1 = { "target-features"="+nontrapping-fptoint" }

; CHECK-LABEL: .custom_section.target_features,"",@

; -atomics, +sign_ext
; ATTRS-NEXT: .int8 3
; ATTRS-NEXT: .int8 45
; ATTRS-NEXT: .int8 7
; ATTRS-NEXT: .ascii "atomics"
; ATTRS-NEXT: .int8 43
; ATTRS-NEXT: .int8 19
; ATTRS-NEXT: .ascii "nontrapping-fptoint"
; ATTRS-NEXT: .int8 43
; ATTRS-NEXT:  int8 8
; ATTRS-NEXT: .ascii "sign-ext"

; -atomics, +simd128
; SIMD128-NEXT: .int8 2
; SIMD128-NEXT: .int8 45
; SIMD128-NEXT: .int8 7
; SIMD128-NEXT: .ascii "atomics"
; SIMD128-NEXT: .int8 43
; SIMD128-NEXT: .int8 7
; SIMD128-NEXT: .ascii "simd128"

; =atomics
; ATOMICS-NEXT: .int8 1
; ATOMICS-NEXT: .int8 61
; ATOMICS-NEXT: .int8 7
; ATOMICS-NEXT: .ascii "atomics"

; =atomics, +nontrapping-fptoint, +sign-ext, +simd128
; BLEEDING-EDGE-NEXT: .int8   4
; BLEEDING-EDGE-NEXT: .int8   61
; BLEEDING-EDGE-NEXT: .int8   7
; BLEEDING-EDGE-NEXT: .ascii  "atomics"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   19
; BLEEDING-EDGE-NEXT: .ascii  "nontrapping-fptoint"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   8
; BLEEDING-EDGE-NEXT: .ascii  "sign-ext"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   7
; BLEEDING-EDGE-NEXT: .ascii  "simd128"

; CHECK-NEXT: .text
