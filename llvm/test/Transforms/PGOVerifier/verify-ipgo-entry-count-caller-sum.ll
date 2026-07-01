; RUN: opt -verify-ipgo -verify-ipgo-print-diagnostics -passes='globaldce' -disable-output %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Validate entry-count-vs-caller-sum behavior:
;  - mismatch is reported when all direct callsite counts are known
;  - exact matches are not reported
;  - multiple direct callers are summed
;  - caller block-frequency cache is used when callsite metadata is missing
;  - unknown direct-callsite counts skip this validation
;  - indirect calls are not counted as direct caller contribution
; Detailed entry-count mismatch assertions are covered in
; llvm/unittests/Transforms/IPO/PGOVerifyTest.cpp.
;
; CHECK: *** IPGO Verification After

@dead_global_for_verify_ipgo_test = internal global i32 42

define dso_local i32 @callee_mismatch(i32 %x) !prof !0 {
entry:
  %t = add i32 %x, 0
  ret i32 %t
}

define dso_local i32 @caller_mismatch(i32 %x) !prof !1 {
entry:
  %r = call i32 @callee_mismatch(i32 %x), !prof !2
  ret i32 %r
}

define dso_local i32 @callee_match(i32 %x) !prof !3 {
entry:
  ret i32 %x
}

define dso_local i32 @caller_match(i32 %x) !prof !4 {
entry:
  %r = call i32 @callee_match(i32 %x), !prof !5
  ret i32 %r
}

define dso_local i32 @callee_multi(i32 %x) !prof !6 {
entry:
  ret i32 %x
}

define dso_local i32 @caller_multi_a(i32 %x) !prof !7 {
entry:
  %r = call i32 @callee_multi(i32 %x), !prof !8
  ret i32 %r
}

define dso_local i32 @caller_multi_b(i32 %x) !prof !9 {
entry:
  %r = call i32 @callee_multi(i32 %x), !prof !10
  ret i32 %r
}

define dso_local i32 @callee_skip_unknown(i32 %x) !prof !11 {
entry:
  ret i32 %x
}

define dso_local i32 @caller_unknown(i32 %x) {
entry:
  ; No !prof on this direct callsite and no function entry count metadata,
  ; so caller-cache fallback cannot provide a known count.
  %r = call i32 @callee_skip_unknown(i32 %x)
  ret i32 %r
}

define dso_local i32 @callee_cache_mismatch(i32 %x) !prof !16 {
entry:
  ret i32 %x
}

define dso_local i32 @caller_cache_known(i32 %x) !prof !17 {
entry:
  ; Missing !prof on callsite, but caller entry count makes block frequency
  ; known and should be used as fallback caller contribution.
  %r = call i32 @callee_cache_mismatch(i32 %x)
  ret i32 %r
}

define dso_local i32 @callee_indirect(i32 %x) !prof !13 {
entry:
  ret i32 %x
}

define dso_local i32 @caller_indirect(ptr %fp, i32 %x) !prof !14 {
entry:
  ; Indirect call profile should not be treated as direct caller contribution.
  %r = call i32 %fp(i32 %x), !prof !15
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"function_entry_count", i64 7}
!2 = !{!"VP", i32 0, i64 7, i64 123456789, i64 7}
!3 = !{!"function_entry_count", i64 7}
!4 = !{!"function_entry_count", i64 7}
!5 = !{!"VP", i32 0, i64 7, i64 223456789, i64 7}
!6 = !{!"function_entry_count", i64 12}
!7 = !{!"function_entry_count", i64 5}
!8 = !{!"VP", i32 0, i64 5, i64 323456789, i64 5}
!9 = !{!"function_entry_count", i64 7}
!10 = !{!"VP", i32 0, i64 7, i64 423456789, i64 7}
!11 = !{!"function_entry_count", i64 10}
!13 = !{!"function_entry_count", i64 9}
!14 = !{!"function_entry_count", i64 9}
!15 = !{!"VP", i32 0, i64 9, i64 523456789, i64 9}
!16 = !{!"function_entry_count", i64 10}
!17 = !{!"function_entry_count", i64 7}
