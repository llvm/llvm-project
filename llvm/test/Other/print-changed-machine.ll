; REQUIRES: aarch64-registered-target
;; --implicit-check-not verifies that analyses passes are not reported.
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed %s 2>&1 | \
; RUN:   FileCheck %s --check-prefixes=VERBOSE,VERBOSE-BAR \
; RUN:     --implicit-check-not='(cseinfo)' --implicit-check-not='Free MachineFunction'
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed -filter-print-funcs=foo %s 2>&1 | FileCheck %s --check-prefixes=VERBOSE,NO-BAR

; VERBOSE:       *** IR Dump After IRTranslator (irtranslator) on foo ***
; VERBOSE-NEXT:  # Machine code for function foo: IsSSA, TracksLiveness{{$}}
; VERBOSE-NEXT:  Function Live Ins: $w0
; VERBOSE-EMPTY:
; VERBOSE-NEXT:  bb.1.entry:
; VERBOSE:       *** IR Dump After AArch64O0PreLegalizerCombiner (aarch64-O0-prelegalizer-combiner) on foo omitted because no change ***
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
; QUIET-FILTER: *** IR Dump After IRTranslator (irtranslator) on atomic_load ***
; QUIET-FILTER: *** IR Dump After IRTranslator (irtranslator) on lr ***

;; dot-cfg/dot-cfg-quiet are unimplemented. Currently they behave like 'quiet'.
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=dot-cfg %s 2>&1 | FileCheck %s --check-prefix=QUIET

;; Covers the IR-level FunctionPasses run by the legacy codegen pass manager.
;; atomic-expand lowers the wide atomic load below before instruction selection.
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=quiet \
; RUN:   -filter-passes=atomic-expand -filter-print-funcs=atomic_load %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=IR-PASS
; IR-PASS:      *** IR Dump After Expand Atomic instructions (atomic-expand) on atomic_load ***
; IR-PASS-NEXT: define i128 @atomic_load(ptr %p) {
; IR-PASS-NEXT:   %1 = cmpxchg ptr %p, i128 0, i128 0 seq_cst seq_cst, align 16

;; Covers the module passes run by MPPassManager; pre-isel-intrinsic-lowering
;; lowers the llvm.load.relative call below.
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=quiet \
; RUN:   -filter-passes=pre-isel-intrinsic-lowering %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MODULE
; MODULE:      *** IR Dump After Pre-ISel Intrinsic Lowering (pre-isel-intrinsic-lowering) on {{.*}} ***
; MODULE:      define ptr @lr(ptr %p, i32 %n) {
; MODULE-NEXT:   %1 = getelementptr i8, ptr %p, i32 %n

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

define i128 @atomic_load(ptr %p) {
  %v = load atomic i128, ptr %p seq_cst, align 16
  ret i128 %v
}

define ptr @lr(ptr %p, i32 %n) {
  %v = call ptr @llvm.load.relative.i32(ptr %p, i32 %n)
  ret ptr %v
}
