; Test hot or unlikely section postfix based on profile and user annotation.
; RUN: llc < %s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: inlinehint norecurse nounwind readnone uwtable
define dso_local i32 @hot1() #0 !prof !31 {
entry:
  ret i32 1
}
; CHECK: .section        .text.hot.,"ax",@progbits
; CHECK: .globl  hot1

; Function Attrs: cold norecurse nounwind readnone uwtable
define dso_local i32 @cold1() #1 !prof !32 {
entry:
  ret i32 1
}
; CHECK: .section        .text.unlikely.,"ax",@progbits
; CHECK: .globl  cold1

; Function Attrs: cold inlinehint noinline norecurse nounwind optsize readnone uwtable
define dso_local i32 @hot2() #2 !prof !31 {
entry:
  ret i32 1
}
; CHECK: .section        .text.hot.,"ax",@progbits
; CHECK: .globl  hot2

define dso_local i32 @normal() {
entry:
  ret i32 1
}
; CHECK: text
; CHECK: .globl  normal

; Function Attrs: hot noinline norecurse nounwind readnone uwtable
define dso_local i32 @hot3() #3 !prof !32 {
entry:
  ret i32 1
}
; CHECK: .section        .text.hot.,"ax",@progbits
; CHECK: .globl  hot3

; Function Attrs: cold noinline norecurse nounwind optsize readnone uwtable
define dso_local i32 @cold2() #4 {
entry:
  ret i32 1
}
; CHECK: .section        .text.unlikely.,"ax",@progbits
; CHECK: .globl  cold2

; Function Attrs: hot noinline norecurse nounwind readnone uwtable
define dso_local i32 @hot4() #3 {
entry:
  ret i32 1
}
; CHECK: .section        .text.hot.,"ax",@progbits
; CHECK: .globl  hot4

attributes #0 = { inlinehint norecurse nounwind readnone uwtable }
attributes #1 = { cold norecurse nounwind readnone uwtable }
attributes #2 = { cold inlinehint noinline norecurse nounwind optsize readnone uwtable }
attributes #3 = { hot noinline norecurse nounwind readnone uwtable }
attributes #4 = { cold noinline norecurse nounwind optsize readnone uwtable }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!30}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 402020}
!5 = !{!"MaxCount", i64 200000}
!6 = !{!"MaxInternalCount", i64 2000}
!7 = !{!"MaxFunctionCount", i64 200000}
!8 = !{!"NumCounts", i64 7}
!9 = !{!"NumFunctions", i64 5}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}
!14 = !{i32 10000, i64 200000, i32 1}
!15 = !{i32 100000, i64 200000, i32 1}
!16 = !{i32 200000, i64 200000, i32 1}
!17 = !{i32 300000, i64 200000, i32 1}
!18 = !{i32 400000, i64 200000, i32 1}
!19 = !{i32 500000, i64 100000, i32 3}
!20 = !{i32 600000, i64 100000, i32 3}
!21 = !{i32 700000, i64 100000, i32 3}
!22 = !{i32 800000, i64 100000, i32 3}
!23 = !{i32 900000, i64 100000, i32 3}
!24 = !{i32 950000, i64 100000, i32 3}
!25 = !{i32 990000, i64 100000, i32 3}
!26 = !{i32 999000, i64 2000, i32 4}
!27 = !{i32 999900, i64 2000, i32 4}
!28 = !{i32 999990, i64 10, i32 6}
!29 = !{i32 999999, i64 10, i32 6}
!30 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git 53c5fdd59a5cf7fbb4dcb7a7e84c9c4a40d32a84)"}
!31 = !{!"function_entry_count", i64 100000}
!32 = !{!"function_entry_count", i64 10}
