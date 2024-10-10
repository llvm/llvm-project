; RUN: opt -passes=deadargelim -S < %s | FileCheck %s

; Check if function level metadatas are properly cloned.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s = common dso_local local_unnamed_addr global i32 0, align 4

define internal i32 @va_func(i32 %num, ...) !prof !28 !PGOFuncName !29{
; CHECK: define internal void @va_func.retelim(i32 %num) !prof ![[ENTRYCOUNT:[0-9]+]] !PGOFuncName ![[PGOFUNCNAME1:[0-9]+]] {
entry:
  %0 = load i32, ptr @s, align 4, !tbaa !31
  %add = add nsw i32 %0, %num
  store i32 %add, ptr @s, align 4, !tbaa !31
  ret i32 0
}

define internal fastcc i32 @foo() unnamed_addr !prof !28 !PGOFuncName !30 {
; CHECK: define internal fastcc void @foo.retelim() unnamed_addr !prof ![[ENTRYCOUNT:[0-9]+]] !PGOFuncName ![[PGOFUNCNAME2:[0-9]+]] {
entry:
  %0 = load i32, ptr @s, align 4, !tbaa !31
  %add = add nsw i32 %0, 8
  store i32 %add, ptr @s, align 4, !tbaa !31
  ret i32 0
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 2}
!5 = !{!"MaxCount", i64 1}
!6 = !{!"MaxInternalCount", i64 0}
!7 = !{!"MaxFunctionCount", i64 1}
!8 = !{!"NumCounts", i64 2}
!9 = !{!"NumFunctions", i64 2}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !17, !18, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27}
!12 = !{i32 10000, i64 0, i32 0}
!13 = !{i32 100000, i64 0, i32 0}
!14 = !{i32 200000, i64 0, i32 0}
!15 = !{i32 300000, i64 0, i32 0}
!16 = !{i32 400000, i64 0, i32 0}
!17 = !{i32 500000, i64 1, i32 2}
!18 = !{i32 600000, i64 1, i32 2}
!19 = !{i32 700000, i64 1, i32 2}
!20 = !{i32 800000, i64 1, i32 2}
!21 = !{i32 900000, i64 1, i32 2}
!22 = !{i32 950000, i64 1, i32 2}
!23 = !{i32 990000, i64 1, i32 2}
!24 = !{i32 999000, i64 1, i32 2}
!25 = !{i32 999900, i64 1, i32 2}
!26 = !{i32 999990, i64 1, i32 2}
!27 = !{i32 999999, i64 1, i32 2}
!28 = !{!"function_entry_count", i64 1}
; CHECK: ![[ENTRYCOUNT]] = !{!"function_entry_count", i64 1}
!29 = !{!"foo.c:va_func"}
; CHECK: ![[PGOFUNCNAME1]] = !{!"foo.c:va_func"}
!30 = !{!"foo.c:foo"}
; CHECK: ![[PGOFUNCNAME2]] = !{!"foo.c:foo"}
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
