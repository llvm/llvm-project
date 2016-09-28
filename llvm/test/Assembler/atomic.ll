; RUN: opt < %s | opt -S | FileCheck %s
; RUN: verify-uselistorder %s
; Basic smoke test for atomic operations.

define void @f(i32* %x) {
  ; CHECK: load atomic i32, i32* %x unordered, align 4
  load atomic i32, i32* %x unordered, align 4
  ; CHECK: load atomic volatile i32, i32* %x singlethread acquire, align 4
  load atomic volatile i32, i32* %x singlethread acquire, align 4
  ; CHECK: load atomic volatile i32, i32* %x syncscope(2) acquire, align 4
  load atomic volatile i32, i32* %x syncscope(2) acquire, align 4
  ; CHECK: store atomic i32 3, i32* %x release, align 4
  store atomic i32 3, i32* %x release, align 4
  ; CHECK: store atomic volatile i32 3, i32* %x singlethread monotonic, align 4
  store atomic volatile i32 3, i32* %x singlethread monotonic, align 4
  ; CHECK: store atomic volatile i32 3, i32* %x syncscope(3) monotonic, align 4
  store atomic volatile i32 3, i32* %x syncscope(3) monotonic, align 4
  ; CHECK: cmpxchg i32* %x, i32 1, i32 0 singlethread monotonic monotonic
  cmpxchg i32* %x, i32 1, i32 0 singlethread monotonic monotonic
  ; CHECK: cmpxchg volatile i32* %x, i32 0, i32 1 acq_rel acquire
  cmpxchg volatile i32* %x, i32 0, i32 1 acq_rel acquire
  ; CHECK: cmpxchg i32* %x, i32 42, i32 0 acq_rel monotonic
  cmpxchg i32* %x, i32 42, i32 0 acq_rel monotonic
  ; CHECK: cmpxchg weak i32* %x, i32 13, i32 0 seq_cst monotonic
  cmpxchg weak i32* %x, i32 13, i32 0 seq_cst monotonic
  ; CHECK: cmpxchg weak i32* %x, i32 13, i32 0 syncscope(4) seq_cst monotonic
  cmpxchg weak i32* %x, i32 13, i32 0 syncscope(4) seq_cst monotonic
  ; CHECK: atomicrmw add i32* %x, i32 10 seq_cst
  atomicrmw add i32* %x, i32 10 seq_cst
  ; CHECK: atomicrmw volatile xchg  i32* %x, i32 10 monotonic
  atomicrmw volatile xchg i32* %x, i32 10 monotonic
  ; CHECK: atomicrmw volatile xchg i32* %x, i32 10 syncscope(5) monotonic
  atomicrmw volatile xchg i32* %x, i32 10 syncscope(5) monotonic
  ; CHECK: fence singlethread release
  fence singlethread release
  ; CHECK: fence seq_cst
  fence seq_cst
  ; CHECK: fence syncscope(6) seq_cst
  fence syncscope(6) seq_cst
  ret void
}
