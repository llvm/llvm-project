; RUN: opt -mtriple=arm-arm-none-eabi -consthoist -S < %s | FileCheck %s
; RUN: opt -mtriple=arm-arm-none-eabi -passes='require<profile-summary>,consthoist' -pgso -S < %s | FileCheck %s -check-prefix=PGSO
; RUN: opt -mtriple=arm-arm-none-eabi -consthoist -pgso=false -S < %s | FileCheck %s -check-prefix=NPGSO

; There are different candidates here for the base constant: 1073876992 and
; 1073876996. But we don't want to see the latter because it results in
; negative offsets.

define void @foo() #0 {
entry:
; CHECK-LABEL: @foo
; CHECK-NOT: [[CONST1:%const_mat[0-9]*]] = add i32 %const, -4
; CHECK-LABEL: @foo_pgso
  %0 = load volatile i32, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %or = or i32 %0, 1
  store volatile i32 %or, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %1 = load volatile i32, i32* inttoptr (i32 1073876996 to i32*), align 4
  %and = and i32 %1, -117506048
  store volatile i32 %and, i32* inttoptr (i32 1073876996 to i32*), align 4
  %2 = load volatile i32, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %and1 = and i32 %2, -17367041
  store volatile i32 %and1, i32* inttoptr (i32 1073876996 to i32*), align 4096
  %3 = load volatile i32, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %and2 = and i32 %3, -262145
  store volatile i32 %and2, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %4 = load volatile i32, i32* inttoptr (i32 1073876996 to i32*), align 4
  %and3 = and i32 %4, -8323073
  store volatile i32 %and3, i32* inttoptr (i32 1073876996 to i32*), align 4
  store volatile i32 10420224, i32* inttoptr (i32 1073877000 to i32*), align 8
  %5 = load volatile i32, i32* inttoptr (i32 1073876996 to i32*), align 4096
  %or4 = or i32 %5, 65536
  store volatile i32 %or4, i32* inttoptr (i32 1073876996 to i32*), align 4096
  %6 = load volatile i32, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %or6.i.i = or i32 %6, 16
  store volatile i32 %or6.i.i, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %7 = load volatile i32, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %and7.i.i = and i32 %7, -4
  store volatile i32 %and7.i.i, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %8 = load volatile i32, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %or8.i.i = or i32 %8, 2
  store volatile i32 %or8.i.i, i32* inttoptr (i32 1073881088 to i32*), align 8192
  ret void
}

attributes #0 = { minsize norecurse nounwind optsize readnone uwtable }

define void @foo_pgso() #1 !prof !14 {
entry:
; PGSO-LABEL: @foo_pgso
; PGSO-NOT: [[CONST2:%const_mat[0-9]*]] = add i32 %const, -4
; NPGSO-LABEL: @foo_pgso
; NPGSO: [[CONST2:%const_mat[0-9]*]] = add i32 %const, -4
  %0 = load volatile i32, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %or = or i32 %0, 1
  store volatile i32 %or, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %1 = load volatile i32, i32* inttoptr (i32 1073876996 to i32*), align 4
  %and = and i32 %1, -117506048
  store volatile i32 %and, i32* inttoptr (i32 1073876996 to i32*), align 4
  %2 = load volatile i32, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %and1 = and i32 %2, -17367041
  store volatile i32 %and1, i32* inttoptr (i32 1073876996 to i32*), align 4096
  %3 = load volatile i32, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %and2 = and i32 %3, -262145
  store volatile i32 %and2, i32* inttoptr (i32 1073876992 to i32*), align 4096
  %4 = load volatile i32, i32* inttoptr (i32 1073876996 to i32*), align 4
  %and3 = and i32 %4, -8323073
  store volatile i32 %and3, i32* inttoptr (i32 1073876996 to i32*), align 4
  store volatile i32 10420224, i32* inttoptr (i32 1073877000 to i32*), align 8
  %5 = load volatile i32, i32* inttoptr (i32 1073876996 to i32*), align 4096
  %or4 = or i32 %5, 65536
  store volatile i32 %or4, i32* inttoptr (i32 1073876996 to i32*), align 4096
  %6 = load volatile i32, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %or6.i.i = or i32 %6, 16
  store volatile i32 %or6.i.i, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %7 = load volatile i32, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %and7.i.i = and i32 %7, -4
  store volatile i32 %and7.i.i, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %8 = load volatile i32, i32* inttoptr (i32 1073881088 to i32*), align 8192
  %or8.i.i = or i32 %8, 2
  store volatile i32 %or8.i.i, i32* inttoptr (i32 1073881088 to i32*), align 8192
  ret void
}

attributes #1 = { norecurse nounwind readnone uwtable }  ; no optsize or minsize

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
