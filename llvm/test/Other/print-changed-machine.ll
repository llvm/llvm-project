; REQUIRES: aarch64-registered-target
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed %s 2>&1 | FileCheck %s --check-prefixes=VERBOSE,VERBOSE-BAR
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed -filter-print-funcs=foo %s 2>&1 | FileCheck %s --check-prefixes=VERBOSE,NO-BAR

; VERBOSE:       *** IR Dump After IRTranslator (irtranslator) on foo ***
; VERBOSE-NEXT:  # Machine code for function foo: IsSSA, TracksLiveness{{$}}
; VERBOSE-NEXT:  Function Live Ins: $w0
; VERBOSE-EMPTY:
; VERBOSE-NEXT:  bb.1.entry:
; VERBOSE:       *** IR Dump After Analysis for ComputingKnownBits (gisel-known-bits) on foo omitted because no change ***
; VERBOSE-NEXT:  *** IR Dump After AArch64O0PreLegalizerCombiner (aarch64-O0-prelegalizer-combiner) on foo omitted because no change ***
; VERBOSE:       *** IR Dump After Legalizer (legalizer) on foo ***
; VERBOSE-NEXT:  # Machine code for function foo: IsSSA, TracksLiveness, Legalized
; VERBOSE-NEXT:  Function Live Ins: $w0
; VERBOSE-EMPTY:
; VERBOSE-NEXT:  bb.1.entry:

; VERBOSE-BAR:   *** IR Dump After IRTranslator (irtranslator) on bar ***
; NO-BAR-NOT:    on bar ***

; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=quiet %s 2>&1 | FileCheck %s --check-prefix=QUIET

; QUIET:         *** IR Dump After IRTranslator (irtranslator) on foo ***
; QUIET-NOT:     ***
; QUIET:         *** IR Dump After Localizer (localizer) on foo ***

; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed -filter-passes=irtranslator,legalizer %s 2>&1 | \
; RUN:   FileCheck %s --check-prefixes=VERBOSE-FILTER
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=quiet -filter-passes=irtranslator %s 2>&1 | \
; RUN:   FileCheck %s --check-prefixes=QUIET-FILTER --implicit-check-not='IR Dump'

; VERBOSE-FILTER:      *** IR Dump After IRTranslator (irtranslator) on foo ***
; VERBOSE-FILTER:      *** IR Dump After AArch64O0PreLegalizerCombiner (aarch64-O0-prelegalizer-combiner) on foo filtered out ***
; VERBOSE-FILTER:      *** IR Dump After Legalizer (legalizer) on foo ***
; VERBOSE-FILTER-NOT:  *** IR Dump After {{.*}} () on

; QUIET-FILTER: *** IR Dump After IRTranslator (irtranslator) on foo ***
; QUIET-FILTER: *** IR Dump After IRTranslator (irtranslator) on bar ***

;; dot-cfg/dot-cfg-quiet are unimplemented. Currently they behave like 'quiet'.
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=dot-cfg %s 2>&1 | FileCheck %s --check-prefix=QUIET

@var = global i32 0

define void @foo(i32 %a) {
entry:
  %b = add i32 %a, 1
  store i32 %b, ptr @var
  ret void
}

define void @bar(i32 %a) {
entry:
  %b = add i32 %a, 2
  store i32 %b, ptr @var
  ret void
}
