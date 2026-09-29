; REQUIRES: x86-registered-target

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %S/Inputs/thinlto_cspgo_bar_use.ll -o %t2.bc
; RUN: llvm-profdata merge %S/Inputs/thinlto_cs.proftext -o %t3.profdata
; RUN: llvm-lto2 run -lto-cspgo-profile-file=%t3.profdata -pgo-instrument-entry=false -save-temps -o %t %t1.bc %t2.bc \
; RUN:   -r=%t1.bc,foo,pl \
; RUN:   -r=%t1.bc,bar,l \
; RUN:   -r=%t1.bc,main,plx \
; RUN:   -r=%t2.bc,bar,pl \
; RUN:   -r=%t2.bc,clobber,pl \
; RUN:   -r=%t2.bc,odd,pl \
; RUN:   -r=%t2.bc,even,pl
; RUN: llvm-dis %t.1.4.opt.bc -o - | FileCheck %s --check-prefix=CSUSE

; CSUSE: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; CSUSE: {{![0-9]+}} = !{i32 1, !"CSProfileSummary", {{![0-9]+}}}
; CSUSE-DAG: {{![0-9]+}} = !{!"branch_weights", i32 100000, i32 0}
; CSUSE-DAG: {{![0-9]+}} = !{!"branch_weights", i32 0, i32 100000}

source_filename = "cspgo.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo() #0 !prof !29 {
entry:
  br label %for.body

for.body:
  %i.06 = phi i32 [ 0, %entry ], [ %add1, %for.body ]
  tail call void @bar(i32 %i.06)
  %add = or i32 %i.06, 1
  tail call void @bar(i32 %add)
  %add1 = add nuw nsw i32 %i.06, 2
  %cmp = icmp ult i32 %add1, 200000
  br i1 %cmp, label %for.body, label %for.end, !prof !30

for.end:
  ret void
}

declare dso_local void @bar(i32)

define dso_local i32 @main() !prof !29 {
entry:
  tail call void @foo()
  ret i32 0
}

attributes #0 = { "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 500002}
!5 = !{!"MaxCount", i64 200000}
!6 = !{!"MaxInternalCount", i64 100000}
!7 = !{!"MaxFunctionCount", i64 200000}
!8 = !{!"NumCounts", i64 6}
!9 = !{!"NumFunctions", i64 4}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27}
!12 = !{i32 10000, i64 200000, i32 1}
!13 = !{i32 100000, i64 200000, i32 1}
!14 = !{i32 200000, i64 200000, i32 1}
!15 = !{i32 300000, i64 200000, i32 1}
!16 = !{i32 400000, i64 200000, i32 1}
!17 = !{i32 500000, i64 100000, i32 4}
!18 = !{i32 600000, i64 100000, i32 4}
!19 = !{i32 700000, i64 100000, i32 4}
!20 = !{i32 800000, i64 100000, i32 4}
!21 = !{i32 900000, i64 100000, i32 4}
!22 = !{i32 950000, i64 100000, i32 4}
!23 = !{i32 990000, i64 100000, i32 4}
!24 = !{i32 999000, i64 100000, i32 4}
!25 = !{i32 999900, i64 100000, i32 4}
!26 = !{i32 999990, i64 100000, i32 4}
!27 = !{i32 999999, i64 1, i32 6}
!29 = !{!"function_entry_count", i64 1}
!30 = !{!"branch_weights", i32 100000, i32 1}
