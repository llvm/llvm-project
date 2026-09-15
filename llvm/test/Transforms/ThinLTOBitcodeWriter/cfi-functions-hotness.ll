; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck %s

; TODO: Check that cfi.functions metadata encodes log2 of counters in the upper 6 bits
; of the linkage operand, including CFG block frequency hotness.
; Note: i8 -2 ((63 << 2) | 2) and i8 126 ((31 << 2) | 2) are unreachable because
; WeakDeclaration (2) requires extern_weak which cannot have profile data.
; i8 -1 ((63 << 2) | 3) and i8 127 ((31 << 2) | 3) are unreachable because
; linkage 3 is unused.

; CHECK: !"f_nocount", i8 0
; CHECK: !"f_entry_count", i8 0
; CHECK: !"f_cfg_hot", i8 0
; CHECK: !"f_zero", i8 0
; CHECK: !"f_one", i8 0
; CHECK: !"f_two", i8 0
; CHECK: !"f_125", i8 1
; CHECK: !"f_128", i8 0
; CHECK: !"f_max", i8 0
; CHECK: !"f_non_canonical_max", i8 1
; CHECK: !"f_hot_attr", i8 0
; CHECK: !"f_non_canonical_hot", i8 1
; CHECK: !"f_unknown", i8 0
; CHECK: !"f_decl", i8 1
; CHECK: !"f_weak_decl", i8 2

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

define void @f_two() "cfi-canonical-jump-table" !prof !7 !type !1 {
  ret void
}

define void @f_125() !prof !11 !type !1 {
  ret void
}

define void @f_128() "cfi-canonical-jump-table" !prof !10 !type !1 {
  ret void
}

define void @f_max() "cfi-canonical-jump-table" !prof !8 !type !1 {
  ret void
}

define void @f_non_canonical_max() !prof !8 !type !1 {
  ret void
}

define void @f_hot_attr() "cfi-canonical-jump-table" hot !type !1 {
  ret void
}

define void @f_non_canonical_hot() hot !type !1 {
  ret void
}

define void @f_unknown() "cfi-canonical-jump-table" !prof !9 !type !1 {
  ret void
}

declare !type !1 void @f_decl()
declare !type !1 extern_weak void @f_weak_decl()

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!1 = !{i32 0, !"typeid1"}
!2 = !{!"function_entry_count", i64 1000}
!3 = !{!"function_entry_count", i64 10}
!4 = !{!"branch_weights", i32 999, i32 1}
!5 = !{!"function_entry_count", i64 0}
!6 = !{!"function_entry_count", i64 1}
!7 = !{!"function_entry_count", i64 2}
!8 = !{!"function_entry_count", i64 -2}
!9 = !{!"function_entry_count", i64 -1}
!10 = !{!"function_entry_count", i64 4294967296}
!11 = !{!"function_entry_count", i64 2147483648}
