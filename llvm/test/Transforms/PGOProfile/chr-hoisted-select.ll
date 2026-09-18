; RUN: opt -passes='require<profile-summary>,function(chr)' -verify-each -S %s | FileCheck %s
; RUN: opt -passes='require<profile-summary>,function(chr,instcombine,simplifycfg)' -verify-each -S %s | FileCheck %s --check-prefix=FOLD

; The outer scope hoists %a to entry because it is the condition of %b.
; The inner scope must not split head at the now-hoisted %a or specialize it
; outside the region guarded by the inner scope's merged branch.
define void @hoisted_insert_point() !prof !12 {
; CHECK-LABEL: define void @hoisted_insert_point()
; CHECK: entry:
; CHECK-NEXT: %a = select i1 false, i1 false, i1 false
; CHECK: br i1 {{.*}}, label %entry.split, label %entry.split.nonchr
; CHECK: head:
; CHECK-NEXT: br i1 false, label %side, label %exit
; CHECK: head.nonchr:
; CHECK-NEXT: %a.nonchr = select i1 false, i1 false, i1 false
entry:
  br label %head
head:
  %a = select i1 false, i1 false, i1 false, !prof !13
  br i1 false, label %side, label %exit, !prof !13
side:
  br label %exit
exit:
  %b = select i1 %a, i1 false, i1 false, !prof !13
  %c = select i1 false, i1 false, i1 false, !prof !13
  ret void
}

; A select other than the insertion point can also be hoisted. Preserve its
; condition: specializing it would change the outer scope's branch condition.
define i1 @hoisted_later_select() !prof !12 {
; FOLD-LABEL: define i1 @hoisted_later_select()
; FOLD-NEXT: entry:
; FOLD-NEXT: ret i1 true
; CHECK-LABEL: define i1 @hoisted_later_select()
; CHECK: entry:
; CHECK-NEXT: %a = select i1 true, i1 true, i1 false
; CHECK: br i1 {{.*}}, label %entry.split, label %entry.split.nonchr
; CHECK: head:
; CHECK-NEXT: %keep = select i1 false, i1 false, i1 false
; CHECK-NEXT: br i1 false, label %side, label %exit
; CHECK: head.nonchr:
; CHECK-NEXT: %keep.nonchr = select i1 false, i1 false, i1 false
; CHECK-NEXT: %a.nonchr = select i1 true, i1 true, i1 false
entry:
  br label %head
head:
  %keep = select i1 false, i1 false, i1 false, !prof !13
  %a = select i1 true, i1 true, i1 false, !prof !13
  br i1 false, label %side, label %exit, !prof !13
side:
  br label %exit
exit:
  %b = select i1 %a, i1 false, i1 false, !prof !13
  %c = select i1 false, i1 false, i1 false, !prof !13
  ret i1 %a
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 1000}
!4 = !{!"MaxCount", i64 1000}
!5 = !{!"MaxInternalCount", i64 1000}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 1}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11}
!11 = !{i32 999999, i64 1000, i32 1}
!12 = !{!"function_entry_count", i64 1000}
!13 = !{!"branch_weights", i32 0, i32 1000}
