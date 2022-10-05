; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 | FileCheck %s --check-prefixes=ALL,SM30
; RUN: llc < %s -march=nvptx64 -mcpu=sm_60 | FileCheck %s --check-prefixes=ALL,SM60
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_30 | %ptxas-verify %if !ptxas-11.0 %{-arch=sm_30%} %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}

; CHECK-LABEL: fadd_double
define void @fadd_double(ptr %0, double %1) {
entry:
  ; SM30: atom.cas.b64
  ; SM60: atom.add.f64
  %2 = atomicrmw fadd ptr %0, double %1 monotonic, align 8
  ret void
}

; CHECK-LABEL: fadd_float
define void @fadd_float(ptr %0, float %1) {
entry:
  ; ALL: atom.add.f32
  %2 = atomicrmw fadd ptr %0, float %1 monotonic, align 4
  ret void
}

; CHECK-LABEL: bitwise_i32
define void @bitwise_i32(ptr %0, i32 %1) {
entry:
  ; ALL: atom.and.b32
  %2 = atomicrmw and ptr %0, i32 %1 monotonic, align 4
  ; ALL: atom.or.b32
  %3 = atomicrmw or ptr %0, i32 %1 monotonic, align 4
  ; ALL: atom.xor.b32
  %4 = atomicrmw xor ptr %0, i32 %1 monotonic, align 4
  ; ALL: atom.exch.b32
  %5 = atomicrmw xchg ptr %0, i32 %1 monotonic, align 4
  ret void
}

; CHECK-LABEL: bitwise_i64
define void @bitwise_i64(ptr %0, i64 %1) {
entry:
  ; SM30: atom.cas.b64
  ; SM60: atom.and.b64
  %2 = atomicrmw and ptr %0, i64 %1 monotonic, align 8
  ; SM30: atom.cas.b64
  ; SM60: atom.or.b64
  %3 = atomicrmw or ptr %0, i64 %1 monotonic, align 8
  ; SM30: atom.cas.b64
  ; SM60: atom.xor.b64
  %4 = atomicrmw xor ptr %0, i64 %1 monotonic, align 8
  ; SM30: atom.cas.b64
  ; SM60: atom.exch.b64
  %5 = atomicrmw xchg ptr %0, i64 %1 monotonic, align 8
  ret void
}

; CHECK-LABEL: minmax_i32
define void @minmax_i32(ptr %0, i32 %1) {
entry:
  ; ALL: atom.min.s32
  %2 = atomicrmw min ptr %0, i32 %1 monotonic, align 4
  ; ALL: atom.max.s32
  %3 = atomicrmw max ptr %0, i32 %1 monotonic, align 4
  ; ALL: atom.min.u32
  %4 = atomicrmw umin ptr %0, i32 %1 monotonic, align 4
  ; ALL: atom.max.u32
  %5 = atomicrmw umax ptr %0, i32 %1 monotonic, align 4
  ret void
}

; CHECK-LABEL: minmax_i64
define void @minmax_i64(ptr %0, i64 %1) {
entry:
  ; SM30: atom.cas.b64
  ; SM60: atom.min.s64
  %2 = atomicrmw min ptr %0, i64 %1 monotonic, align 8
  ; SM30: atom.cas.b64
  ; SM60: atom.max.s64
  %3 = atomicrmw max ptr %0, i64 %1 monotonic, align 8
  ; SM30: atom.cas.b64
  ; SM60: atom.min.u64
  %4 = atomicrmw umin ptr %0, i64 %1 monotonic, align 8
  ; SM30: atom.cas.b64
  ; SM60: atom.max.u64
  %5 = atomicrmw umax ptr %0, i64 %1 monotonic, align 8
  ret void
}

; CHECK-LABEL: bitwise_i8
define void @bitwise_i8(ptr %0, i8 %1) {
entry:
  ; ALL: atom.and.b32
  %2 = atomicrmw and ptr %0, i8 %1 monotonic, align 1
  ; ALL: atom.or.b32
  %3 = atomicrmw or ptr %0, i8 %1 monotonic, align 1
  ; ALL: atom.xor.b32
  %4 = atomicrmw xor ptr %0, i8 %1 monotonic, align 1
  ; ALL: atom.cas.b32
  %5 = atomicrmw xchg ptr %0, i8 %1 monotonic, align 1
  ret void
}

; CHECK-LABEL: minmax_i8
define void @minmax_i8(ptr %0, i8 %1) {
entry:
  ; ALL: atom.cas.b32
  %2 = atomicrmw min ptr %0, i8 %1 monotonic, align 1
  ; ALL: atom.cas.b32
  %3 = atomicrmw max ptr %0, i8 %1 monotonic, align 1
  ; ALL: atom.cas.b32
  %4 = atomicrmw umin ptr %0, i8 %1 monotonic, align 1
  ; ALL: atom.cas.b32
  %5 = atomicrmw umax ptr %0, i8 %1 monotonic, align 1
  ret void
}

; CHECK-LABEL: bitwise_i16
define void @bitwise_i16(ptr %0, i16 %1) {
entry:
  ; ALL: atom.and.b32
  %2 = atomicrmw and ptr %0, i16 %1 monotonic, align 2
  ; ALL: atom.or.b32
  %3 = atomicrmw or ptr %0, i16 %1 monotonic, align 2
  ; ALL: atom.xor.b32
  %4 = atomicrmw xor ptr %0, i16 %1 monotonic, align 2
  ; ALL: atom.cas.b32
  %5 = atomicrmw xchg ptr %0, i16 %1 monotonic, align 2
  ret void
}

; CHECK-LABEL: minmax_i16
define void @minmax_i16(ptr %0, i16 %1) {
entry:
  ; ALL: atom.cas.b32
  %2 = atomicrmw min ptr %0, i16 %1 monotonic, align 2
  ; ALL: atom.cas.b32
  %3 = atomicrmw max ptr %0, i16 %1 monotonic, align 2
  ; ALL: atom.cas.b32
  %4 = atomicrmw umin ptr %0, i16 %1 monotonic, align 2
  ; ALL: atom.cas.b32
  %5 = atomicrmw umax ptr %0, i16 %1 monotonic, align 2
  ret void
}

; TODO: We might still want to test other types, such as i128. Currently the
; backend doesn't support them. Atomic expand only supports expansion to cas of
; the same bitwidth, which means even after expansion, the back end still
; doesn't support the instruction. Here we still put the tests. Remove the
; comment once we have proper support, either from atomic expand or backend.

; define void @bitwise_i128(ptr %0, i128 %1) {
; entry:
;   %2 = atomicrmw and ptr %0, i128 %1 monotonic, align 16
;   %3 = atomicrmw or ptr %0, i128 %1 monotonic, align 16
;   %4 = atomicrmw xor ptr %0, i128 %1 monotonic, align 16
;   %5 = atomicrmw xchg ptr %0, i128 %1 monotonic, align 16
;   ret void
; }

; define void @minmax_i128(ptr %0, i128 %1) {
; entry:
;   %2 = atomicrmw min ptr %0, i128 %1 monotonic, align 16
;   %3 = atomicrmw max ptr %0, i128 %1 monotonic, align 16
;   %4 = atomicrmw umin ptr %0, i128 %1 monotonic, align 16
;   %5 = atomicrmw umax ptr %0, i128 %1 monotonic, align 16
;   ret void
; }
