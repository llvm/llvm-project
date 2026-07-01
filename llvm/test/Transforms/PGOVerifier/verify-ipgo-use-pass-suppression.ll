; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s
; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-merge-function.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts
;
; Ensure copied-metadata diagnostics are suppressed while running inside the
; PGOInstrumentationUse pass itself (IsPGOUsePass=true).

; Two functions intentionally share the same !prof metadata node.
define i32 @f1() !prof !10 {
entry:
  ret i32 1
}

define i32 @f2() !prof !10 {
entry:
  ret i32 2
}

; CHECK-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
; CHECK-NOT: PGOVerify# Copied metadata detected in function

; VERIFY-LABEL: *** IPGO Verification After PGOInstrumentationUse ***
; VERIFY-NOT: PGOVerify# Copied metadata detected in function

!10 = !{!"function_entry_count", i64 1}
