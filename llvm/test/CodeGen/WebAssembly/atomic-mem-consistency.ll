; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+atomics,+sign-ext | FileCheck %s

; Currently all wasm atomic memory access instructions are sequentially
; consistent, so even if LLVM IR specifies weaker orderings than that, we
; should upgrade them to sequential ordering and treat them in the same way.

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
; CHECK-NEXT: .functype store_i32_unordered (i32, i32) -> (){{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_unordered(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p unordered, align 4
  ret void
}

; CHECK-LABEL: store_i32_monotonic:
; CHECK-NEXT: .functype store_i32_monotonic (i32, i32) -> (){{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_monotonic(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p monotonic, align 4
  ret void
}

; CHECK-LABEL: store_i32_release:
; CHECK-NEXT: .functype store_i32_release (i32, i32) -> (){{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_release(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p release, align 4
  ret void
}

; CHECK-LABEL: store_i32_seq_cst:
; CHECK-NEXT: .functype store_i32_seq_cst (i32, i32) -> (){{$}}
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
; CHECK-NEXT: .functype add_i32_monotonic (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_monotonic(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v monotonic
  ret i32 %old
}

; CHECK-LABEL: add_i32_acquire:
; CHECK-NEXT: .functype add_i32_acquire (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_acquire(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v acquire
  ret i32 %old
}

; CHECK-LABEL: add_i32_release:
; CHECK-NEXT: .functype add_i32_release (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_release(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v release
  ret i32 %old
}

; CHECK-LABEL: add_i32_acq_rel:
; CHECK-NEXT: .functype add_i32_acq_rel (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_acq_rel(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v acq_rel
  ret i32 %old
}

; CHECK-LABEL: add_i32_seq_cst:
; CHECK-NEXT: .functype add_i32_seq_cst (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32_seq_cst(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v seq_cst
  ret i32 %old
}

; Ternary RMW instruction: cmpxchg
; The success and failure ordering arguments specify how this cmpxchg
; synchronizes with other atomic operations. Both ordering parameters must be at
; least monotonic, the ordering constraint on failure must be no stronger than
; that on success, and the failure ordering cannot be either release or acq_rel.

; CHECK-LABEL: cmpxchg_i32_monotonic_monotonic:
; CHECK-NEXT: .functype cmpxchg_i32_monotonic_monotonic (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_monotonic_monotonic(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_acquire_monotonic:
; CHECK-NEXT: .functype cmpxchg_i32_acquire_monotonic (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_acquire_monotonic(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new acquire monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_release_monotonic:
; CHECK-NEXT: .functype cmpxchg_i32_release_monotonic (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_release_monotonic(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new release monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_acq_rel_monotonic:
; CHECK-NEXT: .functype cmpxchg_i32_acq_rel_monotonic (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_acq_rel_monotonic(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new acq_rel monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_seq_cst_monotonic:
; CHECK-NEXT: .functype cmpxchg_i32_seq_cst_monotonic (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_seq_cst_monotonic(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new seq_cst monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_acquire_acquire:
; CHECK-NEXT: .functype cmpxchg_i32_acquire_acquire (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_acquire_acquire(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new acquire acquire
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_release_acquire:
; CHECK-NEXT: .functype cmpxchg_i32_release_acquire (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_release_acquire(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new release acquire
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_acq_rel_acquire:
; CHECK-NEXT: .functype cmpxchg_i32_acq_rel_acquire (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_acq_rel_acquire(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new acq_rel acquire
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_seq_cst_acquire:
; CHECK-NEXT: .functype cmpxchg_i32_seq_cst_acquire (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_seq_cst_acquire(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new seq_cst acquire
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_seq_cst_seq_cst:
; CHECK-NEXT: .functype cmpxchg_i32_seq_cst_seq_cst (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_seq_cst_seq_cst(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}
