; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s
; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts

; This test targets a verifier corner case in caller-site frequency derivation.
; The call to @plus1 is in block %callsite, whose incoming paths are:
;  - one predecessor edge with explicit profile metadata (known)
;  - one predecessor edge whose frequency is not derivable (unknown)
;
; The verifier must treat the incoming frequency as unknown (not partial-sum)
; and emit the unavailable-count diagnostic.

source_filename = "pgo-unknown-incoming.c"

define internal i32 @plus1(i32 %x) {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define i32 @main(i32 %x) {
entry:
  %cond = icmp sgt i32 %x, 0
  br i1 %cond, label %knownpred, label %unknownpred

knownpred:
  ; Valid weighted branch: one known incoming contribution to %callsite.
  %k = icmp eq i32 %x, 42
  br i1 %k, label %callsite, label %knownfallthrough, !prof !0

knownfallthrough:
  br label %callsite

unknownpred:
  ; No profile metadata here; incoming for this path stays unknown.
  br label %callsite

callsite:
  %v = call i32 @plus1(i32 7)
  ret i32 %v
}

!0 = !{!"branch_weights", i32 5, i32 1}

; CHECK-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
; CHECK: PGOVerify# Not able to determine Block frequency for main, block entry
; CHECK: PGOVerify# Not able to determine Block frequency for main, block knownpred
; CHECK: PGOVerify# Not able to determine Block frequency for main, block unknownpred

; VERIFY-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
