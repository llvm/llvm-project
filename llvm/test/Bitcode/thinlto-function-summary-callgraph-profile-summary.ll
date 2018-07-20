; Test to check the callgraph in summary when there is PGO
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-dis %t.o
; RUN: cat %t.o.ll | FileCheck %s --check-prefix=DIS

; Make sure the assembler doesn't error when parsing the summary
; RUN: llvm-as %t.o.ll

; Check assembled summary.
; RUN: llvm-dis %t.o.bc -o - | FileCheck %s --check-prefix=DIS

; RUN: opt -module-summary %p/Inputs/thinlto-function-summary-callgraph-profile-summary.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED
; RUN: llvm-dis %t3.thinlto.bc
; RUN: cat %t3.thinlto.ll | FileCheck %s --check-prefix=COMBINED-DIS
; Round trip it through llvm-as
; RUN: cat %t3.thinlto.ll | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=COMBINED-DIS

; Make sure the assembler doesn't error when parsing the combined summary
; RUN: llvm-as %t3.thinlto.ll -o %t3.thinlto.o

; Check assembled combined summary.
; RUN: llvm-dis %t3.thinlto.o -o - | FileCheck %s --check-prefix=COMBINED-DIS


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
; CHECK-NEXT:    <PERMODULE_PROFILE {{.*}} op5=1 op6=3 op7=5 op8=1 op9=2 op10=3 op11=4 op12=1 op13=6 op14=2 op15=3 op16=3 op17=7 op18=2 op19=8 op20=2 op21=25 op22=4/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'hot_functionhot1hot2hot3hot4coldnone1none2none3{{.*}}'

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
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED abbrevid=
; COMBINED-NEXT:    <COMBINED_PROFILE {{.*}} op6=[[HOT1:.*]] op7=3 op8=[[COLD:.*]] op9=1 op10=[[HOT2:.*]] op11=3 op12=[[NONE1:.*]] op13=2 op14=[[HOT3:.*]] op15=3 op16=[[NONE2:.*]] op17=2 op18=[[NONE3:.*]] op19=2/>
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
!15 = !{!"branch_weights", i32 100}

; DIS: ^0 = module: (path: "{{.*}}thinlto-function-summary-callgraph-profile-summary.ll.tmp.o{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (guid: 123)
; DIS: ^2 = gv: (name: "none2") ; guid = 3741006263754194003
; DIS: ^3 = gv: (name: "hot3") ; guid = 5026609803865204483
; DIS: ^4 = gv: (name: "hot2") ; guid = 8117347573235780485
; DIS: ^5 = gv: (name: "hot1") ; guid = 9453975128311291976
; DIS: ^6 = gv: (name: "cold") ; guid = 11668175513417606517
; DIS: ^7 = gv: (name: "hot4") ; guid = 13161834114071272798
; DIS: ^8 = gv: (name: "none3") ; guid = 16213681105727317812
; DIS: ^9 = gv: (name: "hot_function", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 16, calls: ((callee: ^5, hotness: hot), (callee: ^6, hotness: cold), (callee: ^4, hotness: hot), (callee: ^7, hotness: cold), (callee: ^10, hotness: none), (callee: ^3, hotness: hot), (callee: ^2, hotness: none), (callee: ^8, hotness: none), (callee: ^1, hotness: critical))))) ; guid = 17381606045411660303
; DIS: ^10 = gv: (name: "none1") ; guid = 17712061229457633252

; COMBINED-DIS: ^0 = module: (path: "{{.*}}thinlto-function-summary-callgraph-profile-summary.ll.tmp.o", hash: (0, 0, 0, 0, 0))
; COMBINED-DIS: ^1 = module: (path: "{{.*}}thinlto-function-summary-callgraph-profile-summary.ll.tmp2.o", hash: (0, 0, 0, 0, 0))
; COMBINED-DIS: ^2 = gv: (guid: 3741006263754194003, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; COMBINED-DIS: ^3 = gv: (guid: 5026609803865204483, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; COMBINED-DIS: ^4 = gv: (guid: 8117347573235780485, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; COMBINED-DIS: ^5 = gv: (guid: 9453975128311291976, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; COMBINED-DIS: ^6 = gv: (guid: 11668175513417606517, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; COMBINED-DIS: ^7 = gv: (guid: 16213681105727317812, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; COMBINED-DIS: ^8 = gv: (guid: 17381606045411660303, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 16, calls: ((callee: ^5, hotness: hot), (callee: ^6, hotness: cold), (callee: ^4, hotness: hot), (callee: ^9, hotness: none), (callee: ^3, hotness: hot), (callee: ^2, hotness: none), (callee: ^7, hotness: none)))))
; COMBINED-DIS: ^9 = gv: (guid: 17712061229457633252, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
