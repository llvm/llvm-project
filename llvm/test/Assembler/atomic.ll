; RUN: opt < %s | opt -S | FileCheck %s
; RUN: verify-uselistorder %s
; Basic smoke test for atomic operations.

define void @f(i32* %x) {
  ; CHECK: load atomic i32, i32* %x unordered, align 4
  load atomic i32, i32* %x unordered, align 4
  ; CHECK: load atomic volatile i32, i32* %x syncscope("singlethread") acquire, align 4
  load atomic volatile i32, i32* %x syncscope("singlethread") acquire, align 4
  ; CHECK: load atomic volatile i32, i32* %x syncscope("agent") acquire, align 4
  load atomic volatile i32, i32* %x syncscope("agent") acquire, align 4
  ; CHECK: store atomic i32 3, i32* %x release, align 4
  store atomic i32 3, i32* %x release, align 4
  ; CHECK: store atomic volatile i32 3, i32* %x syncscope("singlethread") monotonic, align 4
  store atomic volatile i32 3, i32* %x syncscope("singlethread") monotonic, align 4
  ; CHECK: store atomic volatile i32 3, i32* %x syncscope("workgroup") monotonic, align 4
  store atomic volatile i32 3, i32* %x syncscope("workgroup") monotonic, align 4
  ; CHECK: cmpxchg i32* %x, i32 1, i32 0 syncscope("singlethread") monotonic monotonic
  cmpxchg i32* %x, i32 1, i32 0 syncscope("singlethread") monotonic monotonic
  ; CHECK: cmpxchg i32* %x, i32 1, i32 0 syncscope("workitem") monotonic monotonic
  cmpxchg i32* %x, i32 1, i32 0 syncscope("workitem") monotonic monotonic
  ; CHECK: cmpxchg volatile i32* %x, i32 0, i32 1 acq_rel acquire
  cmpxchg volatile i32* %x, i32 0, i32 1 acq_rel acquire
  ; CHECK: cmpxchg i32* %x, i32 42, i32 0 acq_rel monotonic
  cmpxchg i32* %x, i32 42, i32 0 acq_rel monotonic
  ; CHECK: cmpxchg weak i32* %x, i32 13, i32 0 seq_cst monotonic
  cmpxchg weak i32* %x, i32 13, i32 0 seq_cst monotonic
  ; CHECK: atomicrmw add i32* %x, i32 10 seq_cst
  atomicrmw add i32* %x, i32 10 seq_cst
  ; CHECK: atomicrmw volatile xchg  i32* %x, i32 10 monotonic
  atomicrmw volatile xchg i32* %x, i32 10 monotonic
  ; CHECK: atomicrmw volatile xchg  i32* %x, i32 10 syncscope("agent") monotonic
  atomicrmw volatile xchg i32* %x, i32 10 syncscope("agent") monotonic
  ; CHECK: fence syncscope("singlethread") release
  fence syncscope("singlethread") release
  ; CHECK: fence seq_cst
  fence seq_cst
  ; CHECK: fence syncscope("device") seq_cst
  fence syncscope("device") seq_cst
  ret void
}

define void @fp_atomics(float* %x) {
 ; CHECK: atomicrmw fadd float* %x, float 1.000000e+00 seq_cst
  atomicrmw fadd float* %x, float 1.0 seq_cst

  ; CHECK: atomicrmw volatile fadd float* %x, float 1.000000e+00 seq_cst
  atomicrmw volatile fadd float* %x, float 1.0 seq_cst

  ret void
}
