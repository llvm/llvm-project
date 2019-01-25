; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=DEFAULT-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='lto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='thinlto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='thinlto<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-POSTLINK-Os

; REQUIRES: asserts

; Splitting should occur after Mem2Reg and should be followed by InstCombine.

; DEFAULT-Os: Promote Memory to Register
; DEFAULT-Os: Hot Cold Splitting
; DEFAULT-Os: Combine redundant instructions

; LTO-PRELINK-Os-LABEL: Starting llvm::Module pass manager run.
; LTO-PRELINK-Os: Running pass: {{.*}}PromotePass
; LTO-PRELINK-Os: Running pass: HotColdSplittingPass

; THINLTO-PRELINK-Os-LABEL: Running analysis: PassInstrumentationAnalysis
; THINLTO-PRELINK-Os: Running pass: {{.*}}PromotePass
; THINLTO-PRELINK-Os: Running pass: HotColdSplittingPass

; THINLTO-POSTLINK-Os-NOT: HotColdSplitting
