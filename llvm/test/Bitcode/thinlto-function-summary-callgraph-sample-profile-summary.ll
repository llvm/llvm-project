; Test to check the callgraph in summary when there is PGO
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: opt -module-summary %p/Inputs/thinlto-function-summary-callgraph-sample-profile-summary.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED


; CHECK: <SOURCE_FILENAME
; "hot_function"
; CHECK-NEXT: <FUNCTION op0=0 op1=12
; "hot1"
; CHECK-NEXT: <FUNCTION op0=12 op1=4
; "hot2"
; CHECK-NEXT: <FUNCTION op0=16 op1=4
; "hot3"
; CHECK-NEXT: <FUNCTION op0=20 op1=4
; "hot4"
; CHECK-NEXT: <FUNCTION op0=24 op1=5
; "cold"
; CHECK-NEXT: <FUNCTION op0=29 op1=5
; "none1"
; CHECK-NEXT: <FUNCTION op0=34 op1=5
; "none2"
; CHECK-NEXT: <FUNCTION op0=39 op1=5
; "none3"
; CHECK-NEXT: <FUNCTION op0=44 op1=5
; CHECK-NEXT: <FUNCTION op0=49 op1=5

; CHECK-LABEL:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <VALUE_GUID op0=26 op1=123/>
; op4=none1 op6=hot1 op8=cold1 op10=none2 op12=hot2 op14=cold2 op16=none3 op18=hot3 op20=cold3 op22=123
; CHECK-NEXT:    <PERMODULE_PROFILE {{.*}} op5=7 op6=0 op7=1 op8=3 op9=4 op10=1 op11=8 op12=0 op13=2 op14=3 op15=5 op16=1 op17=9 op18=0 op19=3 op20=3 op21=6 op22=1 op23=26 op24=4/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'hot_functionhot1hot2hot3cold1cold2cold3none1none2none3{{.*}}'

; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
; COMBINED-NEXT:    <FLAGS
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED_PROFILE {{.*}} op6=[[NONE1:.*]] op7=0 op8=[[HOT1:.*]] op9=3 op10=[[COLD1:.*]] op11=1 op12=[[NONE2:.*]] op13=0 op14=[[HOT2:.*]] op15=3 op16=[[COLD2:.*]] op17=1 op18=[[NONE3:.*]] op19=0 op20=[[HOT3:.*]] op21=3 op22=[[COLD3:.*]] op23=1/>
; COMBINED_NEXT:    <COMBINED abbrevid=
; COMBINED_NEXT:  </GLOBALVAL_SUMMARY_BLOCK>


; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This function have high profile count, so entry block is hot.
define void @hot_function(i1 %a, i1 %a2) !prof !20 {
entry:
  call void @none1()
  call void @hot1(), !prof !15
  call void @cold1(), !prof !16
  br i1 %a, label %Cold, label %Hot, !prof !41
Cold:           ; 1/1000 goes here
  call void @none2()
  call void @hot2(), !prof !15
  call void @cold2(), !prof !16
  br label %exit
Hot:            ; 999/1000 goes here
  call void @none3()
  call void @hot3(), !prof !15
  call void @cold3(), !prof !16
  br label %exit
exit:
  ret void
}

declare void @hot1() #1
declare void @hot2() #1
declare void @hot3() #1
declare void @cold1() #1
declare void @cold2() #1
declare void @cold3() #1
declare void @none1() #1
declare void @none2() #1
declare void @none3() #1

!41 = !{!"branch_weights", i32 1, i32 1000}

!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 110, i64 123}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"SampleProfile"}
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
!15 = !{!"branch_weights", i32 100}
!16 = !{!"branch_weights", i32 1}
