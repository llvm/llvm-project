; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='default<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=DEFAULT-Os
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='lto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='thinlto-pre-link<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-PRELINK-Os
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='lto<Os>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-POSTLINK-Os
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -LINK-Os

; REQUIRES: asserts

; Splitting should occur late.

; DEFAULT-Os: pass: HotColdSplittingPass

; LTO-PRELINK-Os-NOT: pass: HotColdSplittingPass

; THINLTO-PRELINK-Os-NOT: Running pass: HotColdSplittingPass

; LTO-POSTLINK-Os: HotColdSplitting
; THINLTO-POSTLINK-Os: HotColdSplitting
