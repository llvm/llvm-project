; RUN: llc < %s | FileCheck %s --check-prefix NOATOMIC
; RUN: not llc < %s -mtriple=wasm32-unknown-unknown -mattr=+atomics,+sign-ext 2>&1 | FileCheck %s --check-prefixes NOEMSCRIPTEN
; RUN: not llc < %s -mtriple=wasm32-unknown-wasi -mattr=+atomics,+sign-ext 2>&1 | FileCheck %s --check-prefixes NOEMSCRIPTEN
; RUN: llc < %s -mtriple=wasm32-unknown-emscripten -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+atomics,+sign-ext | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; NOEMSCRIPTEN: LLVM ERROR: ATOMIC_FENCE is not yet supported in non-emscripten OSes

; A multithread fence turns into 'global.get $__stack_pointer' followed by an
; idempotent atomicrmw instruction.
; CHECK-LABEL: multithread_fence:
; CHECK:      global.get  $push[[SP:[0-9]+]]=, __stack_pointer
; CHECK-NEXT: i32.const $push[[ZERO:[0-9]+]]=, 0
; CHECK-NEXT: i32.atomic.rmw.or  $drop=, 0($pop[[SP]]), $pop[[ZERO]]
; NOATOMIC-NOT: i32.atomic.rmw.or
define void @multithread_fence() {
  fence seq_cst
  ret void
}

; Fences with weaker memory orderings than seq_cst should be treated the same
; because atomic memory access in wasm are sequentially consistent.
; CHECK-LABEL: multithread_weak_fence:
; CHECK:  global.get  $push{{.+}}=, __stack_pointer
; CHECK:  i32.atomic.rmw.or
; CHECK:  i32.atomic.rmw.or
; CHECK:  i32.atomic.rmw.or
define void @multithread_weak_fence() {
  fence acquire
  fence release
  fence acq_rel
  ret void
}

; A singlethread fence becomes compiler_fence instruction, a pseudo instruction
; that acts as a compiler barrier. The barrier should not be emitted to .s file.
; CHECK-LABEL: singlethread_fence:
; CHECK-NOT:  compiler_fence
define void @singlethread_fence() {
  fence syncscope("singlethread") seq_cst
  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") release
  fence syncscope("singlethread") acq_rel
  ret void
}
