; RUN: opt -S -passes='default<O0>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O0 --check-prefix=UNIQUE
; RUN: opt -S -passes='default<O1>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE
; RUN: opt -S -passes='default<O2>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE
; RUN: opt -S -passes='thinlto-pre-link<O1>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE
; RUN: opt -S -passes='thinlto-pre-link<O2>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE

define internal i32 @foo() {
entry:
  ret i32 0
}

define dso_local i32 (...)* @bar() {
entry:
  ret i32 (...)* bitcast (i32 ()* @foo to i32 (...)*)
}

; O0: Running pass: UniqueInternalLinkageNamesPass

;; Check UniqueInternalLinkageNamesPass is scheduled before SampleProfileProbePass.
; O2: Running pass: UniqueInternalLinkageNamesPass
; O2: Running pass: SampleProfileProbePass

; UNIQUE: define internal i32 @foo.__uniq.{{[0-9a-f]+}}()
; UNIQUE: ret {{.*}} @foo.__uniq.{{[0-9a-f]+}} {{.*}}
