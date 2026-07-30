; Generate summary sections
; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto_cspgo_bar.ll -o %t2.o
; RUN: llvm-profdata merge -o %t.profdata %p/Inputs/cspgo.proftext

; RUN: rm -f %t1.o.4.opt.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=cs-profile-path=%t.profdata \
; RUN:    --plugin-opt=jobs=1 \
; RUN:    %t1.o %t2.o -o %t3
; RUN: opt -S %t2.o.4.opt.bc | FileCheck %s

source_filename = "cspgo.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: CSProfileSummary

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

define i32 @main() !prof !29 {
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
!29 = !{!"function_entry_count", i64 1}
!30 = !{!"branch_weights", i32 100000, i32 1}
