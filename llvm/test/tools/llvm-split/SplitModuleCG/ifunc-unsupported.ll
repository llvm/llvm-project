; RUN: not llvm-split -enable-call-graph-split-module=true -j2 -o %t %s 2>&1 | FileCheck %s

; Test the currently unsupported case of a GlobalIFunc whose resolver is
; assigned to a different partition. Ifuncs are always placed in partition 0,
; where the resolver then only appears as a declaration, which the verifier
; rejects ("IFunc resolver must be a definition").
;
; TODO: A follow-up patch will keep an ifunc and its resolver in the same
; partition. Update this test to check the fixed behavior at that point.

; CHECK: IFunc resolver must be a definition

@ifunc_fn = ifunc void (), ptr @resolver

; @resolver is an expensive call-graph root, so the cost-based partitioning
; places it in partition 1, away from its ifunc in partition 0.
define ptr @resolver() {
  ret ptr @target
}

define void @target() {
  ret void
}

define void @root() {
  ret void
}
