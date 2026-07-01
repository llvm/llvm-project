; RUN: opt < %s -verify-ipgo -verify-ipgo-print-diagnostics -verify-ipgo-funcs=ext_callee -passes=instcombine -S -disable-output 2>&1 | FileCheck %s
; REQUIRES: asserts

;
; Verify that -verify-ipgo-funcs still honors verifier exclusions.
; Even when explicitly listed, available_externally functions are skipped.
;
; CHECK: *** IPGO Verification After
; CHECK-NOT: PGOVerify# Entry count mismatch in function ext_callee

define available_externally i32 @ext_callee(i32 %x) !prof !10 {
entry:
  %y = add i32 %x, 0
  ret i32 %y
}

define i32 @main() !prof !11 {
entry:
  %r = call i32 @ext_callee(i32 7), !prof !12
  ret i32 %r
}

; Intentionally mismatched if ext_callee were verified: entry=2, caller-sum=1.
!10 = !{!"function_entry_count", i64 2}
!11 = !{!"function_entry_count", i64 1}
!12 = !{!"VP", i32 0, i64 1, i64 123456789, i64 1}
