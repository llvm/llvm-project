; RUN: opt < %s -disable-output -passes=print-profile-summary -S 2>&1 | FileCheck %s

; The "hot" function attribute is documented (LangRef, and Clang's
; AttributeReference for __attribute__((hot))) to take precedence over profile
; information, so that users can compensate for training input that does not
; cover all of the hot functions. Check that ProfileSummaryInfo honors it.

; An entry count of 0 would otherwise make this function cold.
define void @hot_attr_cold_count() hot !prof !20 {
; CHECK: hot_attr_cold_count :hot entry
  ret void
}

; No entry count at all: the hotness would otherwise be unknown.
define void @hot_attr_no_count() hot {
; CHECK: hot_attr_no_count :hot entry
  ret void
}

; Without the attribute, an entry count of 0 is cold.
define void @plain_cold_count() !prof !20 {
; CHECK: plain_cold_count :cold entry
  ret void
}

; The cold attribute is unaffected.
define void @cold_attr() cold !prof !20 {
; CHECK: cold_attr :cold entry
  ret void
}

; A hot entry count is still hot without the attribute.
define void @plain_hot_count() !prof !21 {
; CHECK: plain_hot_count :hot entry
  ret void
}

!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 0}
!21 = !{!"function_entry_count", i64 400}

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
