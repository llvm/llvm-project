; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=DEFAULT-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='lto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='thinlto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='thinlto<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-POSTLINK-Os

; REQUIRES: asserts

; Splitting should occur late.

; DEFAULT-Os: Hot Cold Splitting
; DEFAULT-Os: Simplify the CFG

; The new pass manager intentionally does not provide a way to differentiate
; between an FullLTO prelink and a non-LTO pipeline. Therefore, expect splitting
; to occur late in the FullLTO prelink and in the postlink.
; LTO-PRELINK-Os-LABEL: Starting llvm::Module pass manager run.
; LTO-PRELINK-Os: Running pass: {{.*}}PromotePass
; LTO-PRELINK-Os: Running pass: HotColdSplittingPass

; THINLTO-PRELINK-Os-LABEL: Running analysis: PassInstrumentationAnalysis
; THINLTO-PRELINK-Os-NOT: Running pass: HotColdSplittingPass

; THINLTO-POSTLINK-Os: HotColdSplitting
