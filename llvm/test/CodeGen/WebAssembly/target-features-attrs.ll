; RUN: llc -mcpu=mvp < %s | FileCheck %s
; RUN: llc -mcpu=mvp -mattr=+simd128 < %s | FileCheck %s --check-prefixes SIMD128

; Test that codegen emits target features from the command line or
; function attributes correctly and that features are enabled for the
; entire module if they are enabled for any function in the module.

target triple = "wasm32-unknown-unknown"

define void @fn_atomics(ptr %p1, float %f2) #0 {
  %a = atomicrmw min ptr undef, i32 42 seq_cst
  %v = fptoui float %f2 to i32
  store i32 %v, ptr %p1
  ret void
}

define void @fn_nontrapping_fptoint(ptr %p1, float %f2) #1 {
  %a = atomicrmw min ptr undef, i32 42 seq_cst
  %v = fptoui float %f2 to i32
  store i32 %v, ptr %p1
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
; CHECK:       loop
; CHECK:       i32.atomic.rmw.cmpxchg
; CHECK:       end_loop

; nontrapping fptoint
; CHECK:       i32.trunc_sat_f32_u
; CHECK:       i32.store

; `fn_nontrapping_fptoint` should be the same as `fn_atomics`
; CHECK-LABEL: fn_nontrapping_fptoint:

; Expanded atomicrmw min
; CHECK:       loop
; CHECK:       i32.atomic.rmw.cmpxchg
; CHECK:       end_loop

; nontrapping fptoint
; CHECK:       i32.trunc_sat_f32_u
; CHECK:       i32.store

; Features in function attributes:
; +atomics, +nontrapping-fptoint, +reference-types
; CHECK-LABEL: .custom_section.target_features,"",@
; CHECK-NEXT: .int8  4
; CHECK-NEXT: .int8  43
; CHECK-NEXT: .int8  7
; CHECK-NEXT: .ascii  "atomics"
; CHECK-NEXT: .int8  43
; CHECK-NEXT: .int8  22
; CHECK-NEXT: .ascii  "call-indirect-overlong"
; CHECK-NEXT: .int8  43
; CHECK-NEXT: .int8  19
; CHECK-NEXT: .ascii  "nontrapping-fptoint"
; CHECK-NEXT: .int8  43
; CHECK-NEXT: .int8  15
; CHECK-NEXT: .ascii  "reference-types"

; Features in function attributes + features specified by -mattr= option:
; +atomics, +nontrapping-fptoint, +reference-types, +simd128
; SIMD128-LABEL: .custom_section.target_features,"",@
; SIMD128-NEXT: .int8  5
; SIMD128-NEXT: .int8  43
; SIMD128-NEXT: .int8  7
; SIMD128-NEXT: .ascii  "atomics"
; SIMD128-NEXT: .int8  43
; SIMD128-NEXT: .int8  22
; SIMD128-NEXT: .ascii  "call-indirect-overlong"
; SIMD128-NEXT: .int8  43
; SIMD128-NEXT: .int8  19
; SIMD128-NEXT: .ascii  "nontrapping-fptoint"
; SIMD128-NEXT: .int8  43
; SIMD128-NEXT: .int8  15
; SIMD128-NEXT: .ascii  "reference-types"
; SIMD128-NEXT: .int8  43
; SIMD128-NEXT: .int8  7
; SIMD128-NEXT: .ascii  "simd128"
