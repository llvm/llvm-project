; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s
; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts

; This test ensures caller-site count derivation remains conservative when all
; incoming paths to a callsite block are unknown.
;
; The callsite block has two predecessors and neither edge has profile metadata.
; The verifier must treat the callsite count as unavailable and emit the
; unavailable-count diagnostic.

source_filename = "pgo-all-unknown-incoming.c"

define internal i32 @plus1(i32 %x) {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define i32 @main(i32 %x) {
entry:
  %cond = icmp sgt i32 %x, 0
  br i1 %cond, label %pred1, label %pred2

pred1:
  br label %callsite

pred2:
  br label %callsite

callsite:
  %v = call i32 @plus1(i32 9)
  ret i32 %v
}

; CHECK-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
; CHECK: PGOVerify# Not able to determine Block frequency for main, block entry
; CHECK: PGOVerify# Not able to determine Block frequency for main, block pred1
; CHECK: PGOVerify# Not able to determine Block frequency for main, block pred2

; VERIFY-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
