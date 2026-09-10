; REQUIRES: asserts
; RUN: split-file %s %t
; RUN: opt -passes=attributor -stats -stats-json -S %t/dead.ll 2>&1 | FileCheck %s --check-prefix=CLEANUP --implicit-check-not="define "
; RUN: opt -passes=attributor -S %t/live.ll | FileCheck %s --check-prefix=LIVE --implicit-check-not="define internal"
;
; With no external entry into the internal cycle, no AAs are seeded during
; deduction. Cleanup only needs the liveness AAs for a and b to inspect uses
; and delete all three functions. It must not seed additional callee AAs.
;
; Without the phase check in assumeLive, initializing b's liveness AA seeds
; additional AAs for a and c. All three functions are deleted either way, so
; checking their deletion alone would not cover the fix.
;
; The expected two AAs are AAIsDeadFunction for a and b. Checking the call to c
; only needs b's liveness AA, since b contains that call.
; Without the phase check, assumeLive(b) calls markLiveInternalFunction for a
; and c, triggering default AA initialization. This can create additional
; function AAs, such as memory behavior and heap-to-stack, as well as AAs for
; a's call to b. The exact additional set depends on the enabled analyses;
; the test checks that only the required liveness AAs are created.
;
; CLEANUP: source_filename =
; CLEANUP: "attributor.NumAAs": 2,
; CLEANUP: "attributor.NumFnDeleted": 3,
;
; The external caller allows normal deduction to optimize away its calls.
; LIVE-LABEL: define void @caller()
; LIVE-NEXT:    ret void
; LIVE-NEXT:  }

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
