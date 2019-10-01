source_filename = "csfdo_bar.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@odd = common dso_local local_unnamed_addr global i32 0, align 4
@even = common dso_local local_unnamed_addr global i32 0, align 4
@not_six = common dso_local local_unnamed_addr global i32 0, align 4

define void @bar(i32 %n) !prof !29 {
entry:
  %call = tail call fastcc i32 @cond(i32 %n)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.else, label %if.then, !prof !30

if.then:
  %0 = load i32, i32* @odd, align 4
  %inc = add i32 %0, 1
  store i32 %inc, i32* @odd, align 4
  br label %for.inc

if.else:
  %1 = load i32, i32* @even, align 4
  %inc1 = add i32 %1, 1
  store i32 %inc1, i32* @even, align 4
  br label %for.inc

for.inc:
  %rem.1 = srem i32 %n, 6
  %tobool2.1 = icmp eq i32 %rem.1, 0
  br i1 %tobool2.1, label %for.inc.1, label %if.then3.1, !prof !35

if.then3.1:
  %2 = load i32, i32* @not_six, align 4
  %inc4.1 = add i32 %2, 1
  store i32 %inc4.1, i32* @not_six, align 4
  br label %for.inc.1

for.inc.1:
  %mul.2 = shl nsw i32 %n, 1
  %rem.2 = srem i32 %mul.2, 6
  %tobool2.2 = icmp eq i32 %rem.2, 0
  br i1 %tobool2.2, label %for.inc.2, label %if.then3.2, !prof !35

if.then3.2:
  %3 = load i32, i32* @not_six, align 4
  %inc4.2 = add i32 %3, 1
  store i32 %inc4.2, i32* @not_six, align 4
  br label %for.inc.2

for.inc.2:
  %mul.3 = mul nsw i32 %n, 3
  %rem.3 = srem i32 %mul.3, 6
  %tobool2.3 = icmp eq i32 %rem.3, 0
  br i1 %tobool2.3, label %for.inc.3, label %if.then3.3, !prof !35

if.then3.3:
  %4 = load i32, i32* @not_six, align 4
  %inc4.3 = add i32 %4, 1
  store i32 %inc4.3, i32* @not_six, align 4
  br label %for.inc.3

for.inc.3:
  ret void
}

define internal fastcc i32 @cond(i32 %i) #1 !prof !29 !PGOFuncName !36 {
entry:
  %rem = srem i32 %i, 2
  ret i32 %rem
}

attributes #1 = { inlinehint noinline }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1700001}
!5 = !{!"MaxCount", i64 800000}
!6 = !{!"MaxInternalCount", i64 399999}
!7 = !{!"MaxFunctionCount", i64 800000}
!8 = !{!"NumCounts", i64 8}
!9 = !{!"NumFunctions", i64 4}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27}
!12 = !{i32 10000, i64 800000, i32 1}
!13 = !{i32 100000, i64 800000, i32 1}
!14 = !{i32 200000, i64 800000, i32 1}
!15 = !{i32 300000, i64 800000, i32 1}
!16 = !{i32 400000, i64 800000, i32 1}
!17 = !{i32 500000, i64 399999, i32 2}
!18 = !{i32 600000, i64 399999, i32 2}
!19 = !{i32 700000, i64 399999, i32 2}
!20 = !{i32 800000, i64 200000, i32 3}
!21 = !{i32 900000, i64 100000, i32 6}
!22 = !{i32 950000, i64 100000, i32 6}
!23 = !{i32 990000, i64 100000, i32 6}
!24 = !{i32 999000, i64 100000, i32 6}
!25 = !{i32 999900, i64 100000, i32 6}
!26 = !{i32 999990, i64 100000, i32 6}
!27 = !{i32 999999, i64 100000, i32 6}
!29 = !{!"function_entry_count", i64 200000}
!30 = !{!"branch_weights", i32 100000, i32 100000}
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !{!"branch_weights", i32 400001, i32 399999}
!36 = !{!"csfdo_bar.c:cond"}
