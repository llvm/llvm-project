; REQUIRES: asserts
; RUN: split-file %s %t
; RUN: opt -passes=attributor -debug-only=attributor -S %t/dead.ll 2>&1 | FileCheck %s --check-prefix=CLEANUP --implicit-check-not="Seeding live internal callee" --implicit-check-not="define "
; RUN: opt -passes=attributor -debug-only=attributor -disable-output %t/live.ll 2>&1 | FileCheck %s --check-prefix=SEED
;
; With no external entry into the internal cycle, deduction does not seed any
; AAs. Cleanup's dead-function check creates liveness AAs while inspecting the
; calls in the cycle. Initializing a live block at this point must not seed
; its internal callees: deduction and manifestation have already finished.
;
; Without the phase check, initializing b's liveness AA seeds a and c during
; cleanup. All three functions are deleted either way, so checking their
; deletion alone would not cover the fix.
;
; CLEANUP: Identified and initialized 0 abstract attributes.
; CLEANUP: Delete/replace at least 0 functions
; CLEANUP: Call site callback failed for {{ *}}call void @b()
; CLEANUP: Deleted 3 functions after manifest.
; CLEANUP: source_filename =
;
; A live external caller still seeds its internal callees during deduction.
; SEED: [AAIsDead] Seeding live internal callee @a from @caller
; SEED: [AAIsDead] Seeding live internal callee @b from @caller
; SEED: Identified and initialized

;--- dead.ll
define internal void @a() {
  call void @b()
  ret void
}

define internal void @b() {
  call void @a()
  call void @c()
  ret void
}

define internal void @c() {
  ret void
}

;--- live.ll
define void @caller() {
  call void @a()
  call void @b()
  ret void
}

define internal void @a() {
  ret void
}

define internal void @b() {
  ret void
}
