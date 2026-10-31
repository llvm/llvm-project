; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='default<O2>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=DEFAULT-O2
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='lto-pre-link<O2>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-PRELINK-O2
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='thinlto-pre-link<O2>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-PRELINK-O2
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='lto<O2>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=LTO-POSTLINK-O2
; RUN: opt -mtriple=x86_64-- -hot-cold-split=true -passes='thinlto<O2>' -debug-pass-manager < %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=THINLTO-POSTLINK-O2

; REQUIRES: asserts

; Splitting should occur late.

; DEFAULT-O2: pass: HotColdSplittingPass

; LTO-PRELINK-O2-NOT: pass: HotColdSplittingPass

; THINLTO-PRELINK-O2-NOT: Running pass: HotColdSplittingPass

; LTO-POSTLINK-O2: HotColdSplitting
; THINLTO-POSTLINK-O2: HotColdSplitting
