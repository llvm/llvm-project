; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts

;
; Verify entry-count mismatch diagnostics are emitted for two internal callees
; when caller-site sum does not match function_entry_count metadata.

define internal i32 @callee_a(i32 %x) !prof !10 {
entry:
  %y = add i32 %x, 0
  ret i32 %y
}

define internal i32 @callee_b(i32 %x) !prof !11 {
entry:
  %y = add i32 %x, 0
  ret i32 %y
}

define i32 @main() !prof !12 {
entry:
  %a = call i32 @callee_a(i32 7)
  %b = call i32 @callee_b(i32 9)
  %s = add i32 %a, %b
  ret i32 %s
}

; CHECK-LABEL: *** IPGO Verification After InstCombinePass ***

; VERIFY-LABEL: *** IPGO Verification After InstCombinePass ***

!llvm.module.flags = !{!30}
!30 = !{i32 1, !"ProfileSummary", !31}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{!"ProfileFormat", !"InstrProf"}
!33 = !{!"TotalCount", i64 6}
!34 = !{!"MaxCount", i64 3}
!35 = !{!"MaxInternalCount", i64 3}
!36 = !{!"MaxFunctionCount", i64 3}
!37 = !{!"NumCounts", i64 3}
!38 = !{!"NumFunctions", i64 3}
!39 = !{!"DetailedSummary", !40}
!40 = !{!41, !42, !43}
!41 = !{i32 10000, i64 3, i32 1}
!42 = !{i32 999000, i64 2, i32 2}
!43 = !{i32 999999, i64 1, i32 3}

; Entry counts intentionally mismatch caller-site sum (=1 each call site in main)
!10 = !{!"function_entry_count", i64 2}
!11 = !{!"function_entry_count", i64 3}
!12 = !{!"function_entry_count", i64 1}
