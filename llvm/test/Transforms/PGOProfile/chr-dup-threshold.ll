; Test case for capping the cloning in CHR.
; RUN: opt < %s -passes='require<profile-summary>,function(chr)' -chr-dup-threshold=2 -S | FileCheck %s

; c sources for the test case.
; extern void foo(int);
; __attribute__((noinline)) void goo(int r, int s, int t) {
;   if ((r & 2) != 0) {
;     if ((s & 2) != 0) {
;       if ((t & 2) != 0) {
;         foo(111);
;       }
;       if ((t & 4) != 0) {
;         foo(112);
;       }
;     }
;     if ((s & 4) != 0) {
;       if ((t & 2) != 0) {
;         foo(121);
;       }
;       if ((t & 4) != 0) {
;         foo(122);
;       }
;     }
;   }
;   if ((r & 4) != 0) {
;     if ((s & 2) != 0) {
;       if ((t & 2) != 0) {
;         foo(211);
;       }
;       if ((t & 4) != 0) {
;         foo(212);
;       }
;     }
;     if ((s & 4) != 0) {
;       if ((t & 2) != 0) {
;         foo(221);
;       }
;       if ((t & 4) != 0) {
;         foo(222);
;       }
;     }
;   }
; }
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @goo(i32 noundef %r, i32 noundef %s, i32 noundef %t) !prof !34 {
entry:
  %and = and i32 %r, 2
  %cmp.not = icmp eq i32 %and, 0
  br i1 %cmp.not, label %if.end24, label %if.then, !prof !35

if.then:
  %and1 = and i32 %s, 2
  %cmp2.not = icmp eq i32 %and1, 0
  br i1 %cmp2.not, label %if.end11, label %if.then3, !prof !35

if.then3:
  %and4 = and i32 %t, 2
  %cmp5.not = icmp eq i32 %and4, 0
  br i1 %cmp5.not, label %if.end, label %if.then6, !prof !35

if.then6:
  tail call void @foo(i32 noundef 111)
  br label %if.end

if.end:
  %and7 = and i32 %t, 4
  %cmp8.not = icmp eq i32 %and7, 0
  br i1 %cmp8.not, label %if.end11, label %if.then9, !prof !35

if.then9:
  tail call void @foo(i32 noundef 112)
  br label %if.end11

if.end11:
  %and12 = and i32 %s, 4
  %cmp13.not = icmp eq i32 %and12, 0
  br i1 %cmp13.not, label %if.end24, label %if.then14, !prof !35

if.then14:
  %and15 = and i32 %t, 2
  %cmp16.not = icmp eq i32 %and15, 0
  br i1 %cmp16.not, label %if.end18, label %if.then17, !prof !35

if.then17:
  tail call void @foo(i32 noundef 121)
  br label %if.end18

if.end18:
  %and19 = and i32 %t, 4
  %cmp20.not = icmp eq i32 %and19, 0
  br i1 %cmp20.not, label %if.end24, label %if.then21, !prof !35

if.then21:
  tail call void @foo(i32 noundef 122)
  br label %if.end24

if.end24:
  %and25 = and i32 %r, 4
  %cmp26.not = icmp eq i32 %and25, 0
  br i1 %cmp26.not, label %if.end52, label %if.then27, !prof !35

if.then27:
  %and28 = and i32 %s, 2
  %cmp29.not = icmp eq i32 %and28, 0
  br i1 %cmp29.not, label %if.end39, label %if.then30, !prof !35

if.then30:
  %and31 = and i32 %t, 2
  %cmp32.not = icmp eq i32 %and31, 0
  br i1 %cmp32.not, label %if.end34, label %if.then33, !prof !35

if.then33:
  tail call void @foo(i32 noundef 211)
  br label %if.end34

if.end34:
  %and35 = and i32 %t, 4
  %cmp36.not = icmp eq i32 %and35, 0
  br i1 %cmp36.not, label %if.end39, label %if.then37, !prof !35

if.then37:
  tail call void @foo(i32 noundef 212)
  br label %if.end39

if.end39:
  %and40 = and i32 %s, 4
  %cmp41.not = icmp eq i32 %and40, 0
  br i1 %cmp41.not, label %if.end52, label %if.then42, !prof !35

if.then42:
  %and43 = and i32 %t, 2
  %cmp44.not = icmp eq i32 %and43, 0
  br i1 %cmp44.not, label %if.end46, label %if.then45, !prof !35

if.then45:
  tail call void @foo(i32 noundef 221)
  br label %if.end46

if.end46:
  %and47 = and i32 %t, 4
  %cmp48.not = icmp eq i32 %and47, 0
  br i1 %cmp48.not, label %if.end52, label %if.then49, !prof !35

if.then49:
  tail call void @foo(i32 noundef 222)
  br label %if.end52

if.end52:
  ret void
}

; CHECK-LABEL: goo
; CHECK-COUNT-3: {{.*}}.split:
; CHECK-NOT: {{.*}}.split:

declare void @foo(i32 noundef)

!llvm.module.flags = !{!4}

!4 = !{i32 1, !"ProfileSummary", !5}
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!6 = !{!"ProfileFormat", !"InstrProf"}
!7 = !{!"TotalCount", i64 2400001}
!8 = !{!"MaxCount", i64 800000}
!9 = !{!"MaxInternalCount", i64 100000}
!10 = !{!"MaxFunctionCount", i64 800000}
!11 = !{!"NumCounts", i64 19}
!12 = !{!"NumFunctions", i64 4}
!13 = !{!"IsPartialProfile", i64 0}
!14 = !{!"PartialProfileRatio", double 0.000000e+00}
!15 = !{!"DetailedSummary", !16}
!16 = !{!17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32}
!17 = !{i32 10000, i64 800000, i32 1}
!18 = !{i32 100000, i64 800000, i32 1}
!19 = !{i32 200000, i64 800000, i32 1}
!20 = !{i32 300000, i64 800000, i32 1}
!21 = !{i32 400000, i64 100000, i32 17}
!22 = !{i32 500000, i64 100000, i32 17}
!23 = !{i32 600000, i64 100000, i32 17}
!24 = !{i32 700000, i64 100000, i32 17}
!25 = !{i32 800000, i64 100000, i32 17}
!26 = !{i32 900000, i64 100000, i32 17}
!27 = !{i32 950000, i64 100000, i32 17}
!28 = !{i32 990000, i64 100000, i32 17}
!29 = !{i32 999000, i64 100000, i32 17}
!30 = !{i32 999900, i64 100000, i32 17}
!31 = !{i32 999990, i64 100000, i32 17}
!32 = !{i32 999999, i64 100000, i32 17}
!34 = !{!"function_entry_count", i64 100000}
!35 = !{!"branch_weights", i32 0, i32 100000}
!36 = !{!"function_entry_count", i64 1}
!37 = !{!"branch_weights", i32 100000, i32 1}
