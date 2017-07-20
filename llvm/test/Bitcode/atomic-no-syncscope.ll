; RUN: llvm-dis -o - %s.bc | FileCheck %s

; Backwards compatibility test: make sure we can process bitcode without
; synchronization scope names encoded in it.

; CHECK: load atomic i32, i32* %x unordered, align 4
; CHECK: load atomic volatile i32, i32* %x syncscope("singlethread") acquire, align 4
; CHECK: store atomic i32 3, i32* %x release, align 4
; CHECK: store atomic volatile i32 3, i32* %x syncscope("singlethread") monotonic, align 4
; CHECK: cmpxchg i32* %x, i32 1, i32 0 syncscope("singlethread") monotonic monotonic
; CHECK: cmpxchg volatile i32* %x, i32 0, i32 1 acq_rel acquire
; CHECK: cmpxchg i32* %x, i32 42, i32 0 acq_rel monotonic
; CHECK: cmpxchg weak i32* %x, i32 13, i32 0 seq_cst monotonic
; CHECK: atomicrmw add i32* %x, i32 10 seq_cst
; CHECK: atomicrmw volatile xchg  i32* %x, i32 10 monotonic
; CHECK: fence syncscope("singlethread") release
; CHECK: fence seq_cst
