; RUN: llc -mtriple=aarch64 -O0 -global-isel -stop-after=irtranslator -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64 -O0 -fast-isel -stop-after=finalize-isel -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64 -O2 -stop-after=finalize-isel -o - %s | FileCheck %s

; !dereferenceable on a load describes the value the load produces, not the
; address it reads from, so it must not set the dereferenceable MMO flag.
; %p carries no dereferenceability of its own, so there is nothing else for the
; flag to come from.

define ptr @load_dereferenceable_md(ptr %p) {
  ; CHECK-LABEL: name: load_dereferenceable_md
  ; CHECK-NOT: dereferenceable
  %v = load ptr, ptr %p, align 8, !dereferenceable !0
  ret ptr %v
}

; With !invariant.load as well, a spurious dereferenceable flag would make
; MachineInstr::isDereferenceableInvariantLoad() -- "will never trap" -- true.

define ptr @load_dereferenceable_invariant_md(ptr %p) {
  ; CHECK-LABEL: name: load_dereferenceable_invariant_md
  ; CHECK-NOT: dereferenceable
  ; CHECK: invariant load
  %v = load ptr, ptr %p, align 8, !dereferenceable !0, !invariant.load !1
  ret ptr %v
}

!0 = !{i64 8}
!1 = !{}
