; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts
;
; Ensure available_externally functions are excluded by shouldVerifyFunction().

; Should be skipped entirely by verifier.
define available_externally i32 @skip_me(i32 %x) !prof !10 {
entry:
  %y = add i32 %x, 0
  ret i32 %y
}

; Local checked function should be considered, but current verifier behavior does
; not emit an entry-count mismatch for this reduced case.
define internal i32 @checked(i32 %x) !prof !11 {
entry:
  %y = add i32 %x, 0
  ret i32 %y
}

; CHECK-LABEL: *** IPGO Verification After
; CHECK-NOT: PGOVerify# Entry count mismatch in function checked
; CHECK-NOT: skip_me

; VERIFY-LABEL: *** IPGO Verification After
; VERIFY-NOT: PGOVerify# Entry count mismatch in function checked
; VERIFY-NOT: skip_me

!llvm.module.flags = !{!30}
!30 = !{i32 1, !"ProfileSummary", !31}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{!"ProfileFormat", !"InstrProf"}
!33 = !{!"TotalCount", i64 7}
!34 = !{!"MaxCount", i64 7}
!35 = !{!"MaxInternalCount", i64 7}
!36 = !{!"MaxFunctionCount", i64 7}
!37 = !{!"NumCounts", i64 1}
!38 = !{!"NumFunctions", i64 1}
!39 = !{!"DetailedSummary", !40}
!40 = !{!41}
!41 = !{i32 10000, i64 7, i32 1}

!10 = !{!"function_entry_count", i64 100}
!11 = !{!"function_entry_count", i64 7}
