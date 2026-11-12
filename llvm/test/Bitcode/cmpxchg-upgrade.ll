; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; cmpxchg-upgrade.ll.bc was produced by running a version of llvm-as from just
; before the IR change on this file.

define void @test(i32* %addr) {
   cmpxchg i32* %addr, i32 42, i32 0 monotonic
; CHECK: cmpxchg ptr %addr, i32 42, i32 0 monotonic monotonic

   cmpxchg i32* %addr, i32 42, i32 0 acquire
; CHECK: cmpxchg ptr %addr, i32 42, i32 0 acquire acquire

   cmpxchg i32* %addr, i32 42, i32 0 release
; CHECK: cmpxchg ptr %addr, i32 42, i32 0 release monotonic

   cmpxchg i32* %addr, i32 42, i32 0 acq_rel
; CHECK: cmpxchg ptr %addr, i32 42, i32 0 acq_rel acquire

   cmpxchg i32* %addr, i32 42, i32 0 seq_cst
; CHECK: cmpxchg ptr %addr, i32 42, i32 0 seq_cst seq_cst

   ret void
}
