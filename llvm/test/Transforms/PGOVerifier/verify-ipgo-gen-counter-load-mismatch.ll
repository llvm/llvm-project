; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-gen -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=pgo-instr-gen -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts
;
; Ensure Gen-phase counter load mismatch is diagnosed when a function loads
; a counter global that belongs to a different function.

@__profc_bar = global i64 0, align 8

define i32 @foo(i32 %x) {
entry:
  %v = load i64, ptr @__profc_bar, align 8
  %y = add i32 %x, 0
  ret i32 %y
}

; CHECK-LABEL: *** IPGO Verification After PGOInstrumentationGen ***
; CHECK-NOT: PGOVerify# Entry count mismatch
; CHECK-NOT: PGOVerify# Block frequency mismatch

; VERIFY-LABEL: *** IPGO Verification After PGOInstrumentationGen ***
; VERIFY-NOT: PGOVerify# Entry count mismatch
; VERIFY-NOT: PGOVerify# Block frequency mismatch
; VERIFY: PGOVerify[CounterLoadMismatch] foo: Counter variable mismatch: loading bar instead of foo
