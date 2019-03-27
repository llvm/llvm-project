; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=DEFAULT-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='lto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='thinlto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='lto<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-POSTLINK-Os
; RUN: opt -mtriple=x86_64-- -Os -hot-cold-split=true -passes='thinlto<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-POSTLINK-Os

; REQUIRES: asserts

; Splitting should occur late.

; DEFAULT-Os: Hot Cold Splitting
; DEFAULT-Os: Simplify the CFG

; LTO-PRELINK-Os-NOT: pass: HotColdSplittingPass

; THINLTO-PRELINK-Os-NOT: Running pass: HotColdSplittingPass

; LTO-POSTLINK-Os: HotColdSplitting
; THINLTO-POSTLINK-Os: HotColdSplitting
