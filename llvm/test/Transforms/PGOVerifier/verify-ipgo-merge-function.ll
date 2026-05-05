; RUN:  opt < %s -verify-ipgo -passes=mergefunc -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY

; VERIFY: *** IPGO Verification After MergeFunctionsPass ***
; VERIFY-NEXT: PGOVerify[EntryCountMismatch] add1: Entry count mismatch: entry=1 vs caller-sum=2

; ModuleID = '../llvm/test/Transforms/PGOVerifier/verify-ipgo-merge-function.ll'
source_filename = "mymergefun.c"

define internal range(i32 -2147483647, -2147483648) i32 @add1(i32 %x) !prof !29 {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define internal range(i32 -2147483647, -2147483648) i32 @plus1(i32 %x)  !prof !29 {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

; Function Attrs: inlinehint
define i32 @main() #0 !prof !29 {
entry:
  %call = call i32 @add1(i32 5)
  %call4 = call i32 @plus1(i32 7)
  %add = add nsw i32 %call, %call4
  ret i32 %add
}

attributes #0 = { inlinehint }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 3}
!4 = !{!"MaxCount", i64 1}
!5 = !{!"MaxInternalCount", i64 0}
!6 = !{!"MaxFunctionCount", i64 1}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"IsPartialProfile", i64 0}
!10 = !{!"PartialProfileRatio", double 0.000000e+00}
!11 = !{!"DetailedSummary", !12}
!12 = !{!13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28}
!13 = !{i32 10000, i64 0, i32 0}
!14 = !{i32 100000, i64 0, i32 0}
!15 = !{i32 200000, i64 0, i32 0}
!16 = !{i32 300000, i64 0, i32 0}
!17 = !{i32 400000, i64 1, i32 3}
!18 = !{i32 500000, i64 1, i32 3}
!19 = !{i32 600000, i64 1, i32 3}
!20 = !{i32 700000, i64 1, i32 3}
!21 = !{i32 800000, i64 1, i32 3}
!22 = !{i32 900000, i64 1, i32 3}
!23 = !{i32 950000, i64 1, i32 3}
!24 = !{i32 990000, i64 1, i32 3}
!25 = !{i32 999000, i64 1, i32 3}
!26 = !{i32 999900, i64 1, i32 3}
!27 = !{i32 999990, i64 1, i32 3}
!28 = !{i32 999999, i64 1, i32 3}
!29 = !{!"function_entry_count", i64 1}
