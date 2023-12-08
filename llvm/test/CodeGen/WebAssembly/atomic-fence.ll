; RUN: llc < %s | FileCheck %s --check-prefix NOATOMIC
; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+atomics | FileCheck %s

target triple = "wasm32-unknown-unknown"

; A multithread fence is lowered to an atomic.fence instruction.
; CHECK-LABEL: multithread_fence:
; CHECK:  atomic.fence
; NOATOMIC-NOT: i32.atomic.rmw.or
define void @multithread_fence() {
  fence seq_cst
  ret void
}

; Fences with weaker memory orderings than seq_cst should be treated the same
; because atomic memory access in wasm are sequentially consistent.
; CHECK-LABEL: multithread_weak_fence:
; CHECK:       atomic.fence
; CHECK-NEXT:  atomic.fence
; CHECK-NEXT:  atomic.fence
define void @multithread_weak_fence() {
  fence acquire
  fence release
  fence acq_rel
  ret void
}

; A singlethread fence becomes compiler_fence instruction, a pseudo instruction
; that acts as a compiler barrier. The barrier should not be emitted to .s file.
; CHECK-LABEL: singlethread_fence:
; CHECK-NOT: compiler_fence
; CHECK-NOT: atomic_fence
define void @singlethread_fence() {
  fence syncscope("singlethread") seq_cst
  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") release
  fence syncscope("singlethread") acq_rel
  ret void
}
