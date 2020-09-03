; RUN: llc < %s | FileCheck %s --check-prefixes CHECK,ATTRS
; RUN: llc < %s -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -mcpu=bleeding-edge | FileCheck %s --check-prefixes CHECK,BLEEDING-EDGE

; Test that codegen emits target features from the command line or
; function attributes correctly and that features are enabled for the
; entire module if they are enabled for any function in the module.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @fn_atomics(i32* %p1, float %f2) #0 {
  %a = atomicrmw min i32* undef, i32 42 seq_cst
  %v = fptoui float %f2 to i32
  store i32 %v, i32* %p1
  ret void
}

define void @fn_nontrapping_fptoint(i32* %p1, float %f2) #1 {
  %a = atomicrmw min i32* undef, i32 42 seq_cst
  %v = fptoui float %f2 to i32
  store i32 %v, i32* %p1
  ret void
}

define void @fn_reference_types() #2 {
  ret void
}

attributes #0 = { "target-features"="+atomics" }
attributes #1 = { "target-features"="+nontrapping-fptoint" }
attributes #2 = { "target-features"="+reference-types" }

; CHECK-LABEL: fn_atomics:

; Expanded atomicrmw min
; ATTRS:       loop
; CHECK:       i32.atomic.rmw.cmpxchg
; ATTRS:       end_loop

; nontrapping fptoint
; CHECK:       i32.trunc_sat_f32_u
; ATTRS:       i32.store

; `fn_nontrapping_fptoint` should be the same as `fn_atomics`
; CHECK-LABEL: fn_nontrapping_fptoint:

; Expanded atomicrmw min
; ATTRS:       loop
; CHECK:       i32.atomic.rmw.cmpxchg
; ATTRS:       end_loop

; nontrapping fptoint
; CHECK:       i32.trunc_sat_f32_u
; ATTRS:       i32.store

; CHECK-LABEL: .custom_section.target_features,"",@

; +atomics, +nontrapping-fptoint, +reference-types
; ATTRS-NEXT: .int8 3
; ATTRS-NEXT: .int8 43
; ATTRS-NEXT: .int8 7
; ATTRS-NEXT: .ascii "atomics"
; ATTRS-NEXT: .int8 43
; ATTRS-NEXT: .int8 19
; ATTRS-NEXT: .ascii "nontrapping-fptoint"
; ATTRS-NEXT: .int8 43
; ATTRS-NEXT: .int8 15
; ATTRS-NEXT: .ascii "reference-types"

; +atomics, +nontrapping-fptoint, +reference-types, +simd128
; SIMD128-NEXT: .int8 4
; SIMD128-NEXT: .int8 43
; SIMD128-NEXT: .int8 7
; SIMD128-NEXT: .ascii "atomics"
; SIMD128-NEXT: .int8 43
; SIMD128-NEXT: .int8 19
; SIMD128-NEXT: .ascii "nontrapping-fptoint"
; SIMD128-NEXT: .int8 43
; SIMD128-NEXT: .int8 15
; SIMD128-NEXT: .ascii "reference-types"
; SIMD128-NEXT: .int8 43
; SIMD128-NEXT: .int8 7
; SIMD128-NEXT: .ascii "simd128"

; +atomics, +bulk-memory, +mutable-globals, +nontrapping-fptoint,
; +reference-types, +sign-ext, +simd128, +tail-call
; BLEEDING-EDGE-NEXT: .int8   8
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   7
; BLEEDING-EDGE-NEXT: .ascii  "atomics"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   11
; BLEEDING-EDGE-NEXT: .ascii  "bulk-memory"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   15
; BLEEDING-EDGE-NEXT: .ascii  "mutable-globals"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   19
; BLEEDING-EDGE-NEXT: .ascii  "nontrapping-fptoint"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   15
; BLEEDING-EDGE-NEXT: .ascii  "reference-types"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   8
; BLEEDING-EDGE-NEXT: .ascii  "sign-ext"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   7
; BLEEDING-EDGE-NEXT: .ascii  "simd128"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   9
; BLEEDING-EDGE-NEXT: .ascii  "tail-call"

; CHECK-NEXT: .text
