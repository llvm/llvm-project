; Test to check the callgraph in summary when there is PGO
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: opt -module-summary %p/Inputs/thinlto-function-summary-callgraph-profile-summary.ll -o %t2.o
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
; CHECK-NEXT: <FUNCTION op0=24 op1=4
; "cold"
; CHECK-NEXT: <FUNCTION op0=28 op1=4
; "none1"
; CHECK-NEXT: <FUNCTION op0=32 op1=5
; "none2"
; CHECK-NEXT: <FUNCTION op0=37 op1=5
; "none3"
; CHECK-NEXT: <FUNCTION op0=42 op1=5
; CHECK-LABEL:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <VALUE_GUID op0=25 op1=123/>
; op4=hot1 op6=cold op8=hot2 op10=hot4 op12=none1 op14=hot3 op16=none2 op18=none3 op20=123
; CHECK-NEXT:    <PERMODULE_PROFILE {{.*}} op4=1 op5=3 op6=5 op7=1 op8=2 op9=3 op10=4 op11=3 op12=6 op13=2 op14=3 op15=3 op16=7 op17=2 op18=8 op19=2 op20=25 op21=3/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'hot_functionhot1hot2hot3hot4coldnone1none2none3'

; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
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
; COMBINED-NEXT:    <COMBINED_PROFILE {{.*}} op5=[[HOT1:.*]] op6=3 op7=[[COLD:.*]] op8=1 op9=[[HOT2:.*]] op10=3 op11=[[NONE1:.*]] op12=2 op13=[[HOT3:.*]] op14=3 op15=[[NONE2:.*]] op16=2 op17=[[NONE3:.*]] op18=2/>
; COMBINED_NEXT:    <COMBINED abbrevid=
; COMBINED_NEXT:  </GLOBALVAL_SUMMARY_BLOCK>


; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This function have high profile count, so entry block is hot.
define void @hot_function(i1 %a, i1 %a2) !prof !20 {
entry:
    call void @hot1()
    br i1 %a, label %Cold, label %Hot, !prof !41
Cold:           ; 1/1000 goes here
  call void @cold()
  call void @hot2()
  call void @hot4(), !prof !15
  call void @none1()
  br label %exit
Hot:            ; 999/1000 goes here
  call void @hot2()
  call void @hot3()
  br i1 %a2, label %None1, label %None2, !prof !42
None1:          ; half goes here
  call void @none1()
  call void @none2()
  br label %exit
None2:          ; half goes here
  call void @none3()
  br label %exit
exit:
  ret void
}

declare void @hot1() #1
declare void @hot2() #1
declare void @hot3() #1
declare void @hot4() #1
declare void @cold() #1
declare void @none1() #1
declare void @none2() #1
declare void @none3() #1


!41 = !{!"branch_weights", i32 1, i32 1000}
!42 = !{!"branch_weights", i32 1, i32 1}



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
