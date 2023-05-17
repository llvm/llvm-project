;; Test that MemProfContextDisambiguation is enabled under the expected conditions
;; and in the expected position.

;; Pass is not currently enabled by default at any opt level.
; RUN: opt -debug-pass-manager -passes='lto<O0>' -S %s \
; RUN:     2>&1 | FileCheck %s --implicit-check-not="Running pass: MemProfContextDisambiguation"
; RUN: opt -debug-pass-manager -passes='lto<O1>' -S %s \
; RUN:     2>&1 | FileCheck %s --implicit-check-not="Running pass: MemProfContextDisambiguation"
; RUN: opt -debug-pass-manager -passes='lto<O2>' -S %s \
; RUN:     2>&1 | FileCheck %s --implicit-check-not="Running pass: MemProfContextDisambiguation"
; RUN: opt -debug-pass-manager -passes='lto<O3>' -S %s \
; RUN:     2>&1 | FileCheck %s --implicit-check-not="Running pass: MemProfContextDisambiguation"

;; Pass should not run even under option at O0/O1.
; RUN: opt -debug-pass-manager -passes='lto<O0>' -S %s \
; RUN:     -enable-memprof-context-disambiguation \
; RUN:     2>&1 | FileCheck %s --implicit-check-not="Running pass: MemProfContextDisambiguation"
; RUN: opt -debug-pass-manager -passes='lto<O1>' -S %s \
; RUN:     -enable-memprof-context-disambiguation \
; RUN:     2>&1 | FileCheck %s --implicit-check-not="Running pass: MemProfContextDisambiguation"

;; Pass should be enabled under option at O2/O3.
; RUN: opt -debug-pass-manager -passes='lto<O2>' -S %s \
; RUN:     -enable-memprof-context-disambiguation \
; RUN:     2>&1 | FileCheck %s --check-prefix=ENABLED
; RUN: opt -debug-pass-manager -passes='lto<O3>' -S %s \
; RUN:     -enable-memprof-context-disambiguation \
; RUN:     2>&1 | FileCheck %s --check-prefix=ENABLED

;; When enabled, MemProfContextDisambiguation runs just after inlining.
; ENABLED: Running pass: InlinerPass
; ENABLED: Invalidating analysis: InlineAdvisorAnalysis
; ENABLED: Running pass: MemProfContextDisambiguation

define noundef ptr @_Z3barv() {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10)
  ret ptr %call
}

declare noundef nonnull ptr @_Znam(i64 noundef)
