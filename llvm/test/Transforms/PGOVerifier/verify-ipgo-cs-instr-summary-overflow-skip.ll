; REQUIRES: asserts
; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
;
; Coverage test for hasInstrProfUseSummary:
; - Uses CS profile summary (queried first via getProfileSummary(true)).
; - Current behavior does not emit unknown block-frequency diagnostics for this IR.

define internal i32 @cs_summary_loop(i32 %n) !prof !10 {
entry:
  %x = add i32 %n, 0
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %cmp = icmp slt i32 %i, %x
  br i1 %cmp, label %body, label %exit

body:
  br label %latch

latch:
  %inc = add i32 %i, 1
  br label %header

exit:
  ret i32 %i
}

; CHECK-LABEL: *** IPGO Verification After InstCombinePass ***
; CHECK-NOT: PGOVerify# Not able to determine Block frequency for cs_summary_loop, block header

; VERIFY-LABEL: *** IPGO Verification After InstCombinePass ***
; VERIFY-NOT: PGOVerify# Not able to determine Block frequency for cs_summary_loop, block header

!llvm.module.flags = !{!30}
!30 = !{i32 1, !"CSProfileSummary", !31}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{!"ProfileFormat", !"CSInstrProf"}
!33 = !{!"TotalCount", i64 4294967297}
!34 = !{!"MaxCount", i64 4294967296}
!35 = !{!"MaxInternalCount", i64 4294967296}
!36 = !{!"MaxFunctionCount", i64 4294967296}
!37 = !{!"NumCounts", i64 1}
!38 = !{!"NumFunctions", i64 1}
!39 = !{!"DetailedSummary", !40}
!40 = !{!41, !42, !43}
!41 = !{i32 10000, i64 4294967296, i32 1}
!42 = !{i32 999000, i64 4294967296, i32 1}
!43 = !{i32 999999, i64 1, i32 1}

!10 = !{!"function_entry_count", i64 1}
