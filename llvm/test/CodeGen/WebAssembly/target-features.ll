; RUN: llc < %s | FileCheck %s --check-prefixes CHECK,ATTRS
; RUN: llc < %s -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -mcpu=bleeding-edge | FileCheck %s --check-prefixes CHECK,BLEEDING-EDGE

; Test that codegen emits target features from the command line or
; function attributes correctly and that features are enabled for the
; entire module if they are enabled for any function in the module.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @foo(i32* %p1) #0 {
  %a = atomicrmw min i32* undef, i32 42 seq_cst
  %v = fptoui float undef to i32
  store i32 %v, i32* %p1
  ret void
}

define void @bar(i32* %p1) #1 {
  %a = atomicrmw min i32* undef, i32 42 seq_cst
  %v = fptoui float undef to i32
  store i32 %v, i32* %p1
  ret void
}

attributes #0 = { "target-features"="+atomics" }
attributes #1 = { "target-features"="+nontrapping-fptoint" }


; CHECK-LABEL: foo:

; Expanded atomicrmw min
; ATTRS:       loop
; ATTRS:       i32.atomic.rmw.cmpxchg
; SIMD128-NOT: i32.atomic.rmw.cmpxchg
; ATTRS:       end_loop

; nontrapping fptoint
; ATTRS:       i32.trunc_sat_f32_u
; SIMD128-NOT: i32.trunc_sat_f32_u
; ATTRS:       i32.store

; `bar` should be the same as `foo`
; CHECK-LABEL: bar:

; Expanded atomicrmw min
; ATTRS:       loop
; ATTRS:       i32.atomic.rmw.cmpxchg
; SIMD128-NOT: i32.atomic.rmw.cmpxchg
; ATTRS:       end_loop

; nontrapping fptoint
; ATTRS:       i32.trunc_sat_f32_u
; SIMD128-NOT: i32.trunc_sat_f32_u
; ATTRS:       i32.store

; CHECK-LABEL: .custom_section.target_features,"",@

; +atomics, +nontrapping-fptoint
; ATTRS-NEXT: .int8 2
; ATTRS-NEXT: .int8 43
; ATTRS-NEXT: .int8 7
; ATTRS-NEXT: .ascii "atomics"
; ATTRS-NEXT: .int8 43
; ATTRS-NEXT: .int8 19
; ATTRS-NEXT: .ascii "nontrapping-fptoint"

; -atomics, +simd128
; SIMD128-NEXT: .int8 2
; SIMD128-NEXT: .int8 45
; SIMD128-NEXT: .int8 7
; SIMD128-NEXT: .ascii "atomics"
; SIMD128-NEXT: .int8 43
; SIMD128-NEXT: .int8 7
; SIMD128-NEXT: .ascii "simd128"

; +atomics, +nontrapping-fptoint, +sign-ext, +simd128
; BLEEDING-EDGE-NEXT: .int8   5
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   7
; BLEEDING-EDGE-NEXT: .ascii  "atomics"
; BLEEDING-EDGE-NEXT: .int8   43
; BLEEDING-EDGE-NEXT: .int8   15
; BLEEDING-EDGE-NEXT: .ascii  "mutable-globals"
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
