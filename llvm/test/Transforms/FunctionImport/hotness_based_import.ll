; Test to check the callgraph in summary when there is PGO
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/hotness_based_import.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Test import with default hot multiplier (3)
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=1 --S | FileCheck %s --check-prefix=CHECK --check-prefix=HOT-DEFAULT
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=1 --S -import-hot-multiplier=3.0 | FileCheck %s --check-prefix=CHECK --check-prefix=HOT-DEFAULT
; HOT-DEFAULT-DAG: define available_externally void @hot1()
; HOT-DEFAULT-DAG: define available_externally void @hot2()
; HOT-DEFAULT-DAG: define available_externally void @cold()
; HOT-DEFAULT-DAG: define available_externally void @none1()

; HOT-DEFAULT-NOT: define available_externally void @hot3()
; HOT-DEFAULT-NOT: define available_externally void @none2()
; HOT-DEFAULT-NOT: define available_externally void @none3()
; HOT-DEFAULT-NOT: define available_externally void @cold2()


; Test import with hot multiplier 1.0 - treat hot callsites as normal.
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=1 -import-hot-multiplier=1.0 --S | FileCheck %s --check-prefix=CHECK --check-prefix=HOT-ONE
; HOT-ONE-DAG: define available_externally void @hot1()
; HOT-ONE-DAG: define available_externally void @cold()
; HOT-ONE-DAG: define available_externally void @none1()
; HOT-ONE-NOT: define available_externally void @hot2()
; HOT-ONE-NOT: define available_externally void @hot3()
; HOT-ONE-NOT: define available_externally void @none2()
; HOT-ONE-NOT: define available_externally void @none3()
; HOT-ONE-NOT: define available_externally void @cold2()


; Test import with hot multiplier 0.0 and high threshold - don't import functions called from hot callsite.
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=10 -import-hot-multiplier=0.0 --S | FileCheck %s --check-prefix=CHECK --check-prefix=HOT-ZERO
; HOT-ZERO-DAG: define available_externally void @cold()
; HOT-ZERO-DAG: define available_externally void @none1()
; HOT-ZERO-DAG: define available_externally void @none2()
; HOT-ZERO-DAG: define available_externally void @none3()
; HOT-ZERO-DAG: define available_externally void @cold2()
; HOT-ZERO-NOT: define available_externally void @hot2()
; HOT-ZERO-NOT: define available_externally void @hot1()
; HOT-ZERO-NOT: define available_externally void @hot3()



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
  call void @cold2()
  call void @hot2()
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
declare void @cold() #1
declare void @cold2() #1
declare void @none1() #1
declare void @none2() #1
declare void @none3() #1


!41 = !{!"branch_weights", i32 1, i32 1000}
!42 = !{!"branch_weights", i32 1, i32 1}



!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 110}

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
