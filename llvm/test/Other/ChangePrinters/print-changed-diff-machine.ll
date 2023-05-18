; REQUIRES: aarch64-registered-target
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=diff %s 2>&1 | FileCheck %s --check-prefixes=DIFF,VERBOSE
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=diff-quiet %s 2>&1 | FileCheck %s --check-prefixes=DIFF,QUIET
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=cdiff %s 2>&1 | FileCheck %s --check-prefixes=CDIFF,VERBOSE
; RUN: llc -filetype=null -mtriple=aarch64 -O0 -print-changed=cdiff-quiet %s 2>&1 | FileCheck %s --check-prefixes=CDIFF,QUIET

; VERBOSE:    *** IR Dump After AArch64O0PreLegalizerCombiner (aarch64-O0-prelegalizer-combiner) on foo omitted because no change ***
; QUIET-NOT:  *** {{.*}} omitted because no change ***

; DIFF:      *** IR Dump After Legalizer (legalizer) on foo ***
; DIFF-NEXT: -# Machine code for function foo: IsSSA, TracksLiveness
; DIFF-NEXT: +# Machine code for function foo: IsSSA, TracksLiveness, Legalized
; DIFF-NEXT:  Function Live Ins: $w0

; CDIFF:      *** IR Dump After Legalizer (legalizer) on foo ***
; CDIFF-NEXT: {{.\[31m-}}# Machine code for function foo: IsSSA, TracksLiveness{{.\[0m}}
; CDIFF-NEXT: {{.\[32m\+}}# Machine code for function foo: IsSSA, TracksLiveness, Legalized{{.\[0m}}

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
