; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts
;
; Exercise recursive entry-count mismatch warning path.
; Caller-site sum (from recursive callsite) is intentionally less than entry count.

define internal i32 @rec(i32 %n) !prof !10 {
entry:
  %cond = icmp sgt i32 %n, 0
  br i1 %cond, label %recurse, label %base, !prof !11

recurse:
  %n1 = sub nsw i32 %n, 1
  %r = call i32 @rec(i32 %n1)
  ret i32 %r

base:
  ret i32 0
}

; CHECK-LABEL: *** IPGO Verification After InstCombinePass ***

; VERIFY-LABEL: *** IPGO Verification After InstCombinePass ***

!llvm.module.flags = !{!30}
!30 = !{i32 1, !"ProfileSummary", !31}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{!"ProfileFormat", !"InstrProf"}
!33 = !{!"TotalCount", i64 10}
!34 = !{!"MaxCount", i64 10}
!35 = !{!"MaxInternalCount", i64 10}
!36 = !{!"MaxFunctionCount", i64 10}
!37 = !{!"NumCounts", i64 3}
!38 = !{!"NumFunctions", i64 1}
!39 = !{!"DetailedSummary", !40}
!40 = !{!41}
!41 = !{i32 10000, i64 10, i32 1}

!10 = !{!"function_entry_count", i64 10}
; entry -> recurse:3, entry -> base:7
!11 = !{!"branch_weights", i32 3, i32 7}
