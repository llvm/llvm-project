; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

; Cold function, %dup should not be duplicated into predecessors.
define i32 @cold(i32 %a, i32* %p, i32* %q) !prof !21 {
; CHECK-LABEL: cold
; CHECK:       %entry
; CHECK:       %true1
; CHECK:       %dup
; CHECK:       %true2
; CHECK:       %false1
; CHECK:       %false2
entry:
  %cond1 = icmp sgt i32 %a, 1
  br i1 %cond1, label %true1, label %false1, !prof !30

true1:
  %v1 = load i32, i32* %p, align 4
  %v2 = add i32 %v1, 2
  br label %dup

false1:
  %v3 = load i32, i32* %q, align 4
  %v4 = sub i32 %v3, 3
  br label %dup

dup:
  %v5 = phi i32 [%v2, %true1], [%v4, %false1]
  %cond2 = icmp sgt i32 %v5, 4
  br i1 %cond2, label %true2, label %false2, !prof !30

true2:
  %v6 = xor i32 %v5, %a
  br label %exit

false2:
  %v7 = and i32 %v5, %a
  br label %exit

exit:
  %v8 = phi i32 [%v6, %true2], [%v7, %false2]
  ret i32 %v8
}

; Same code as previous function, but with hot profile count.
; So %dup should be duplicated into predecessors.
define i32 @hot(i32 %a, i32* %p, i32* %q) !prof !22 {
; CHECK-LABEL: hot
; CHECK:       %entry
; CHECK:       %true1
; CHECK:       %false2
; CHECK:       %false1
; CHECK:       %true2
entry:
  %cond1 = icmp sgt i32 %a, 1
  br i1 %cond1, label %true1, label %false1, !prof !30

true1:
  %v1 = load i32, i32* %p, align 4
  %v2 = add i32 %v1, 2
  br label %dup

false1:
  %v3 = load i32, i32* %q, align 4
  %v4 = sub i32 %v3, 3
  br label %dup

dup:
  %v5 = phi i32 [%v2, %true1], [%v4, %false1]
  %cond2 = icmp sgt i32 %v5, 4
  br i1 %cond2, label %true2, label %false2, !prof !30

true2:
  %v6 = xor i32 %v5, %a
  br label %exit

false2:
  %v7 = and i32 %v5, %a
  br label %exit

exit:
  %v8 = phi i32 [%v6, %true2], [%v7, %false2]
  ret i32 %v8
}


!llvm.module.flags = !{!1}
!21 = !{!"function_entry_count", i64 10}
!22 = !{!"function_entry_count", i64 400}

!30 = !{!"branch_weights", i32 1, i32 1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
