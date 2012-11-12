; RUN: opt < %s | opt -S | FileCheck %s
; Basic smoke test for atomic operations.

define void @f(i32* %x) {
  ; CHECK: load atomic i32* %x unordered, align 4
  load atomic i32* %x unordered, align 4
  ; CHECK: load atomic volatile i32* %x singlethread acquire, align 4
  load atomic volatile i32* %x singlethread acquire, align 4
  ; CHECK: store atomic i32 3, i32* %x release, align 4
  store atomic i32 3, i32* %x release, align 4
  ; CHECK: store atomic volatile i32 3, i32* %x singlethread monotonic, align 4
  store atomic volatile i32 3, i32* %x singlethread monotonic, align 4
  ; CHECK: cmpxchg i32* %x, i32 1, i32 0 singlethread monotonic
  cmpxchg i32* %x, i32 1, i32 0 singlethread monotonic
  ; CHECK: cmpxchg volatile i32* %x, i32 0, i32 1 acq_rel
  cmpxchg volatile i32* %x, i32 0, i32 1 acq_rel
  ; CHECK: atomicrmw add i32* %x, i32 10 seq_cst
  atomicrmw add i32* %x, i32 10 seq_cst
  ; CHECK: atomicrmw volatile xchg  i32* %x, i32 10 monotonic
  atomicrmw volatile xchg i32* %x, i32 10 monotonic
  ; CHECK: fence singlethread release
  fence singlethread release
  ; CHECK: fence seq_cst
  fence seq_cst
  ret void
}
