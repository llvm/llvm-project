; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel < %s | FileCheck %s

; The IR-specified alignment on atomicrmw / cmpxchg must be carried into the
; MachineMemOperand, matching the atomic load/store paths in
; SelectionDAGBuilder (which use I.getAlign()).  Previously visitAtomicRMW and
; visitAtomicCmpXchg used getEVTAlign(MemVT) (the natural type alignment),
; silently discarding an over-aligned `align N`.

define i32 @rmw_align(ptr %p) {
  ; CHECK-LABEL: name: rmw_align
  ; CHECK: LXADD32 {{.*}} :: (load store seq_cst (s32) on %ir.p, align 32)
  %r = atomicrmw add ptr %p, i32 1 seq_cst, align 32
  ret i32 %r
}

define i32 @cas_align(ptr %p, i32 %c, i32 %n) {
  ; CHECK-LABEL: name: cas_align
  ; CHECK: LCMPXCHG32 {{.*}} :: (load store seq_cst seq_cst (s32) on %ir.p, align 32)
  %r = cmpxchg ptr %p, i32 %c, i32 %n seq_cst seq_cst, align 32
  %v = extractvalue { i32, i1 } %r, 0
  ret i32 %v
}
