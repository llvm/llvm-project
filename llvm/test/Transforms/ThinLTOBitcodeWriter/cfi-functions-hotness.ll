; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck %s

; Check that cfi.functions metadata encodes hotness in bits 2-3 of
; the linkage operand:
; - Hot functions (Attribute::Hot or PSI.isFunctionHotInCallGraph) -> Hot (3)
; - Cold functions (Attribute::Cold or PSI.isFunctionColdInCallGraph) -> Cold (1)
; - Normal functions -> Other (2)
; - Unprofiled functions or declarations -> Unknown (0)

; CHECK: !"f_nocount", i8 0,
; CHECK: !"f_entry_count", i8 12
; CHECK: !"f_cfg_hot", i8 12
; CHECK: !"f_zero", i8 4
; CHECK: !"f_one", i8 4
; CHECK: !"f_hot_attr", i8 12
; CHECK: !"f_cold_attr", i8 4
; CHECK: !"f_non_canonical_hot", i8 13
; CHECK: !"f_non_canonical_cold", i8 5
; CHECK: !"f_non_canonical_nocount", i8 1
; CHECK: !"f_other", i8 8
; CHECK: !"f_non_canonical_other", i8 9
; CHECK: !"f_decl", i8 1,
; CHECK: !"f_weak_decl", i8 2,

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f_nocount() "cfi-canonical-jump-table" !type !1 {
  ret void
}

define void @f_entry_count() "cfi-canonical-jump-table" !prof !2 !type !1 {
  ret void
}

define void @f_cfg_hot(i32 %n) "cfi-canonical-jump-table" !prof !3 !type !1 {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, %n
  br i1 %cond, label %loop, label %exit, !prof !4

exit:
  ret void
}

define void @f_zero() "cfi-canonical-jump-table" !prof !5 !type !1 {
  ret void
}

define void @f_one() "cfi-canonical-jump-table" !prof !6 !type !1 {
  ret void
}

define void @f_hot_attr() "cfi-canonical-jump-table" hot !type !1 {
  ret void
}

define void @f_cold_attr() "cfi-canonical-jump-table" cold !type !1 {
  ret void
}

define void @f_non_canonical_hot() hot !type !1 {
  ret void
}

define void @f_non_canonical_cold() cold !type !1 {
  ret void
}

define void @f_non_canonical_nocount() !type !1 {
  ret void
}

define void @f_other() "cfi-canonical-jump-table" !prof !7 !type !1 {
  ret void
}

define void @f_non_canonical_other() !prof !7 !type !1 {
  ret void
}

declare !type !1 void @f_decl()
declare !type !1 extern_weak void @f_weak_decl()

!llvm.module.flags = !{!0, !20}

!0 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!1 = !{i32 0, !"typeid1"}
!2 = !{!"function_entry_count", i64 1000}
!3 = !{!"function_entry_count", i64 10}
!4 = !{!"branch_weights", i32 999, i32 1}
!5 = !{!"function_entry_count", i64 0}
!6 = !{!"function_entry_count", i64 1}
!7 = !{!"function_entry_count", i64 10}

!20 = !{i32 1, !"ProfileSummary", !21}
!21 = !{!22, !23, !24, !25, !26, !27, !28, !34, !35, !29}
!22 = !{!"ProfileFormat", !"SampleProfile"}
!23 = !{!"TotalCount", i64 10000}
!24 = !{!"MaxCount", i64 10000}
!25 = !{!"MaxInternalCount", i64 1}
!26 = !{!"MaxFunctionCount", i64 1000}
!27 = !{!"NumCounts", i64 3}
!28 = !{!"NumFunctions", i64 3}
!34 = !{!"IsPartialProfile", i64 1}
!35 = !{!"PartialProfileRatio", double 5.000000e-01}
!29 = !{!"DetailedSummary", !30}
!30 = !{!31, !32, !33}
!31 = !{i32 10000, i64 100, i32 1}
!32 = !{i32 999000, i64 100, i32 1}
!33 = !{i32 999999, i64 1, i32 2}
