; RUN: not llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=+atomics,+sign-ext | FileCheck %s

; Currently all wasm atomic memory access instructions are sequentially
; consistent, so even if LLVM IR specifies weaker orderings than that, we
; should upgrade them to sequential ordering and treat them in the same way.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Atomic loads
;===----------------------------------------------------------------------------

; The 'release' and 'acq_rel' orderings are not valid on load instructions.

; CHECK-LABEL: load_i32_unordered:
; CHECK: i32.atomic.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @load_i32_unordered(i32 *%p) {
  %v = load atomic i32, i32* %p unordered, align 4
  ret i32 %v
}

; CHECK-LABEL: load_i32_monotonic:
; CHECK: i32.atomic.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @load_i32_monotonic(i32 *%p) {
  %v = load atomic i32, i32* %p monotonic, align 4
  ret i32 %v
}

; CHECK-LABEL: load_i32_acquire:
; CHECK: i32.atomic.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @load_i32_acquire(i32 *%p) {
  %v = load atomic i32, i32* %p acquire, align 4
  ret i32 %v
}

; CHECK-LABEL: load_i32_seq_cst:
; CHECK: i32.atomic.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @load_i32_seq_cst(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 4
  ret i32 %v
}

;===----------------------------------------------------------------------------
; Atomic stores
;===----------------------------------------------------------------------------

; The 'acquire' and 'acq_rel' orderings aren’t valid on store instructions.

; CHECK-LABEL: store_i32_unordered:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_unordered(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p unordered, align 4
  ret void
}

; CHECK-LABEL: store_i32_monotonic:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_monotonic(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p monotonic, align 4
  ret void
}

; CHECK-LABEL: store_i32_release:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_release(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p release, align 4
  ret void
}

; CHECK-LABEL: store_i32_seq_cst:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_seq_cst(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p seq_cst, align 4
  ret void
}

;===----------------------------------------------------------------------------
; Atomic read-modify-writes
;===----------------------------------------------------------------------------

; Out of several binary RMW instructions, here we test 'add' as an example.
; The 'unordered' ordering is not valid on atomicrmw instructions.

; CHECK-LABEL: add_i32_monotonic:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_monotonic(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v monotonic
  ret i32 %old
}

; CHECK-LABEL: add_i32_acquire:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_acquire(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v acquire
  ret i32 %old
}

; CHECK-LABEL: add_i32_release:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_release(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v release
  ret i32 %old
}

; CHECK-LABEL: add_i32_acq_rel:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_acq_rel(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v acq_rel
  ret i32 %old
}

; CHECK-LABEL: add_i32_seq_cst:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_seq_cst(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v seq_cst
  ret i32 %old
}
