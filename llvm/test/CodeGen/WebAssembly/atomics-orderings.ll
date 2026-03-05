; RUN: llc < %s --mtriple=wasm32 -mattr=+atomics,+relaxed-atomics | FileCheck %s

; CHECK-LABEL: load_i32_acquire:
; CHECK: i32.atomic.load acqrel 0
define i32 @load_i32_acquire(ptr %p) {
  %v = load atomic i32, ptr %p acquire, align 4
  ret i32 %v
}

; CHECK-LABEL: load_i32_seq_cst:
; CHECK: i32.atomic.load seqcst 0
define i32 @load_i32_seq_cst(ptr %p) {
  %v = load atomic i32, ptr %p seq_cst, align 4
  ret i32 %v
}

; CHECK-LABEL: store_i32_release:
; CHECK: i32.atomic.store acqrel 0
define void @store_i32_release(ptr %p, i32 %v) {
  store atomic i32 %v, ptr %p release, align 4
  ret void
}

; CHECK-LABEL: store_i32_seq_cst:
; CHECK: i32.atomic.store seqcst 0
define void @store_i32_seq_cst(ptr %p, i32 %v) {
  store atomic i32 %v, ptr %p seq_cst, align 4
  ret void
}

; CHECK-LABEL: add_i32_acq_rel:
; CHECK: i32.atomic.rmw.add acqrel 0
define i32 @add_i32_acq_rel(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v acq_rel
  ret i32 %old
}

; CHECK-LABEL: add_i32_seq_cst:
; CHECK: i32.atomic.rmw.add seqcst 0
define i32 @add_i32_seq_cst(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_acquire_acquire:
; CHECK: i32.atomic.rmw.cmpxchg acqrel 0
define i32 @cmpxchg_i32_acquire_acquire(ptr %p, i32 %exp, i32 %new) {
  %pair = cmpxchg ptr %p, i32 %exp, i32 %new acquire acquire
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; CHECK-LABEL: cmpxchg_i32_seq_cst_seq_cst:
; CHECK: i32.atomic.rmw.cmpxchg seqcst 0
define i32 @cmpxchg_i32_seq_cst_seq_cst(ptr %p, i32 %exp, i32 %new) {
  %pair = cmpxchg ptr %p, i32 %exp, i32 %new seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; CHECK-LABEL: fence_acquire:
; CHECK: atomic.fence acqrel
define void @fence_acquire() {
  fence acquire
  ret void
}

; CHECK-LABEL: fence_seq_cst:
; CHECK: atomic.fence seqcst
define void @fence_seq_cst() {
  fence seq_cst
  ret void
}

; CHECK-LABEL: load_i32_monotonic:
; CHECK: i32.atomic.load acqrel 0
define i32 @load_i32_monotonic(ptr %p) {
  %v = load atomic i32, ptr %p monotonic, align 4
  ret i32 %v
}

; CHECK-LABEL: store_i32_monotonic:
; CHECK: i32.atomic.store acqrel 0
define void @store_i32_monotonic(ptr %p, i32 %v) {
  store atomic i32 %v, ptr %p monotonic, align 4
  ret void
}

; CHECK-LABEL: add_i32_release:
; CHECK: i32.atomic.rmw.add acqrel 0
define i32 @add_i32_release(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v release
  ret i32 %old
}

; CHECK-LABEL: add_i32_acquire:
; CHECK: i32.atomic.rmw.add acqrel 0
define i32 @add_i32_acquire(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v acquire
  ret i32 %old
}

; CHECK-LABEL: add_i32_monotonic:
; CHECK: i32.atomic.rmw.add acqrel 0
define i32 @add_i32_monotonic(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v monotonic
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_acq_rel_monotonic:
; CHECK: i32.atomic.rmw.cmpxchg acqrel 0
define i32 @cmpxchg_i32_acq_rel_monotonic(ptr %p, i32 %exp, i32 %new) {
  %pair = cmpxchg ptr %p, i32 %exp, i32 %new acq_rel monotonic
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; CHECK-LABEL: cmpxchg_i32_release_monotonic:
; CHECK: i32.atomic.rmw.cmpxchg acqrel 0
define i32 @cmpxchg_i32_release_monotonic(ptr %p, i32 %exp, i32 %new) {
  %pair = cmpxchg ptr %p, i32 %exp, i32 %new release monotonic
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; CHECK-LABEL: cmpxchg_i32_monotonic_monotonic:
; CHECK: i32.atomic.rmw.cmpxchg acqrel 0
define i32 @cmpxchg_i32_monotonic_monotonic(ptr %p, i32 %exp, i32 %new) {
  %pair = cmpxchg ptr %p, i32 %exp, i32 %new monotonic monotonic
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; CHECK-LABEL: fence_release:
; CHECK: atomic.fence acqrel
define void @fence_release() {
  fence release
  ret void
}

; CHECK-LABEL: fence_acq_rel:
; CHECK: atomic.fence acqrel
define void @fence_acq_rel() {
  fence acq_rel
  ret void
}


