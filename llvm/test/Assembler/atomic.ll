; RUN: opt < %s | opt -S | FileCheck %s
; RUN: verify-uselistorder %s
; Basic smoke test for atomic operations.

define void @f(ptr %x) {
  ; CHECK: load atomic i32, ptr %x unordered, align 4
  load atomic i32, ptr %x unordered, align 4
  ; CHECK: load atomic volatile i32, ptr %x syncscope("singlethread") acquire, align 4
  load atomic volatile i32, ptr %x syncscope("singlethread") acquire, align 4
  ; CHECK: load atomic volatile i32, ptr %x syncscope("agent") acquire, align 4
  load atomic volatile i32, ptr %x syncscope("agent") acquire, align 4
  ; CHECK: store atomic i32 3, ptr %x release, align 4
  store atomic i32 3, ptr %x release, align 4
  ; CHECK: store atomic volatile i32 3, ptr %x syncscope("singlethread") monotonic, align 4
  store atomic volatile i32 3, ptr %x syncscope("singlethread") monotonic, align 4
  ; CHECK: store atomic volatile i32 3, ptr %x syncscope("workgroup") monotonic, align 4
  store atomic volatile i32 3, ptr %x syncscope("workgroup") monotonic, align 4
  ; CHECK: cmpxchg ptr %x, i32 1, i32 0 syncscope("singlethread") monotonic monotonic
  cmpxchg ptr %x, i32 1, i32 0 syncscope("singlethread") monotonic monotonic
  ; CHECK: cmpxchg ptr %x, i32 1, i32 0 syncscope("workitem") monotonic monotonic
  cmpxchg ptr %x, i32 1, i32 0 syncscope("workitem") monotonic monotonic
  ; CHECK: cmpxchg volatile ptr %x, i32 0, i32 1 acq_rel acquire
  cmpxchg volatile ptr %x, i32 0, i32 1 acq_rel acquire
  ; CHECK: cmpxchg ptr %x, i32 42, i32 0 acq_rel monotonic
  cmpxchg ptr %x, i32 42, i32 0 acq_rel monotonic
  ; CHECK: cmpxchg weak ptr %x, i32 13, i32 0 seq_cst monotonic
  cmpxchg weak ptr %x, i32 13, i32 0 seq_cst monotonic
  ; CHECK: atomicrmw add ptr %x, i32 10 seq_cst
  atomicrmw add ptr %x, i32 10 seq_cst
  ; CHECK: atomicrmw volatile xchg  ptr %x, i32 10 monotonic
  atomicrmw volatile xchg ptr %x, i32 10 monotonic
  ; CHECK: atomicrmw volatile xchg  ptr %x, i32 10 syncscope("agent") monotonic
  atomicrmw volatile xchg ptr %x, i32 10 syncscope("agent") monotonic

  ; CHECK: atomicrmw volatile uinc_wrap ptr %x, i32 10 monotonic
  atomicrmw volatile uinc_wrap ptr %x, i32 10 monotonic
  ; CHECK: atomicrmw volatile uinc_wrap ptr %x, i32 10 syncscope("agent") monotonic
  atomicrmw volatile uinc_wrap ptr %x, i32 10 syncscope("agent") monotonic

  ; CHECK: atomicrmw volatile udec_wrap ptr %x, i32 10 monotonic
  atomicrmw volatile udec_wrap ptr %x, i32 10 monotonic
  ; CHECK: atomicrmw volatile udec_wrap ptr %x, i32 10 syncscope("agent") monotonic
  atomicrmw volatile udec_wrap ptr %x, i32 10 syncscope("agent") monotonic

  ; CHECK: atomicrmw volatile usub_cond ptr %x, i32 10 monotonic
  atomicrmw volatile usub_cond ptr %x, i32 10 monotonic
  ; CHECK: atomicrmw volatile usub_cond ptr %x, i32 10 syncscope("agent") monotonic
  atomicrmw volatile usub_cond ptr %x, i32 10 syncscope("agent") monotonic

  ; CHECK: atomicrmw volatile usub_sat ptr %x, i32 10 monotonic
  atomicrmw volatile usub_sat ptr %x, i32 10 monotonic
  ; CHECK: atomicrmw volatile usub_sat ptr %x, i32 10 syncscope("agent") monotonic
  atomicrmw volatile usub_sat ptr %x, i32 10 syncscope("agent") monotonic

  ; CHECK: fence syncscope("singlethread") release
  fence syncscope("singlethread") release
  ; CHECK: fence seq_cst
  fence seq_cst
  ; CHECK: fence syncscope("device") seq_cst
  fence syncscope("device") seq_cst
  ret void
}

define void @fp_atomics(ptr %x) {
 ; CHECK: atomicrmw fadd ptr %x, float 1.000000e+00 seq_cst
  atomicrmw fadd ptr %x, float 1.0 seq_cst

  ; CHECK: atomicrmw volatile fadd ptr %x, float 1.000000e+00 seq_cst
  atomicrmw volatile fadd ptr %x, float 1.0 seq_cst

  ; CHECK: atomicrmw fmax ptr %x, float 1.000000e+00 seq_cst
  atomicrmw fmax ptr %x, float 1.0 seq_cst

  ; CHECK: atomicrmw volatile fmax ptr %x, float 1.000000e+00 seq_cst
  atomicrmw volatile fmax ptr %x, float 1.0 seq_cst

  ; CHECK: atomicrmw fmin ptr %x, float 1.000000e+00 seq_cst
  atomicrmw fmin ptr %x, float 1.0 seq_cst

  ; CHECK: atomicrmw volatile fmin ptr %x, float 1.000000e+00 seq_cst
  atomicrmw volatile fmin ptr %x, float 1.0 seq_cst

  ret void
}

define void @fp_vector_atomicrmw(ptr %x, <2 x half> %val) {
  ; CHECK: %atomic.fadd = atomicrmw fadd ptr %x, <2 x half> %val seq_cst
  %atomic.fadd = atomicrmw fadd ptr %x, <2 x half> %val seq_cst

  ; CHECK: %atomic.fsub = atomicrmw fsub ptr %x, <2 x half> %val seq_cst
  %atomic.fsub = atomicrmw fsub ptr %x, <2 x half> %val seq_cst

  ; CHECK: %atomic.fmax = atomicrmw fmax ptr %x, <2 x half> %val seq_cst
  %atomic.fmax = atomicrmw fmax ptr %x, <2 x half> %val seq_cst

  ; CHECK: %atomic.fmin = atomicrmw fmin ptr %x, <2 x half> %val seq_cst
  %atomic.fmin = atomicrmw fmin ptr %x, <2 x half> %val seq_cst

  ret void
}
