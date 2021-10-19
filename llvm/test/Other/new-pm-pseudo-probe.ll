; RUN: opt -S -passes='default<O1>' -new-pm-pseudo-probe-for-profiling -debug-pass-manager < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes='default<O2>' -new-pm-pseudo-probe-for-profiling -debug-pass-manager < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes='thinlto-pre-link<O1>' -new-pm-pseudo-probe-for-profiling -debug-pass-manager < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes='thinlto-pre-link<O2>' -new-pm-pseudo-probe-for-profiling -debug-pass-manager < %s 2>&1 | FileCheck %s

define void @foo() {
  ret void
}

;; Check the SampleProfileProbePass is enabled under the -new-pm-pseudo-probe-for-profiling switch.
;; The switch can be used to test a specific pass order in a particular setup, e.g, in unique-internal-linkage-names.ll
; CHECK: Running pass: SampleProfileProbePass
