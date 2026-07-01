; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s
; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pass-remarks-analysis=verify-ipgo -S -disable-output 2>&1 | FileCheck %s --check-prefix=REMARK
; REQUIRES: asserts

; This test validates that pgo-instr-use + verify-ipgo pipeline remains clean
; for this small profile input without emitting entry-count mismatch diagnostics.

source_filename = "strict_profile_counts.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 @callee(i32 %x) {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define i32 @main() {
entry:
  %v = call i32 @callee(i32 42)
  ret i32 %v
}

; CHECK-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
; CHECK-NOT: PGOVerify# Entry count mismatch in function

; VERIFY-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
; VERIFY-NOT: PGOVerify# Entry count mismatch in function

; REMARK-NOT: remark:
