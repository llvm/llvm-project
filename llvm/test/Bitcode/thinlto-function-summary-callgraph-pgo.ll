; Test to check the callgraph in summary when there is PGO
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; RUN: opt -module-summary %p/Inputs/thinlto-function-summary-callgraph.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED

; Check parsing for old summary versions generated from this file.
; RUN: llvm-lto -thinlto-index-stats %p/Inputs/thinlto-function-summary-callgraph-pgo.1.bc  | FileCheck %s --check-prefix=OLD
; RUN: llvm-lto -thinlto-index-stats %p/Inputs/thinlto-function-summary-callgraph-pgo-combined.1.bc  | FileCheck %s --check-prefix=OLD-COMBINED

; CHECK: <SOURCE_FILENAME
; CHECK-NEXT: <FUNCTION
; "func"
; CHECK-NEXT: <FUNCTION op0=4 op1=4
; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <FLAGS
; See if the call to func is registered, using the expected hotness type.
; CHECK-NEXT:    <PERMODULE_PROFILE {{.*}} op7=1 op8=2/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>
; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'mainfunc{{.*}}'

; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
; COMBINED-NEXT:    <FLAGS
; COMBINED-NEXT:    <VALUE_GUID op0=[[FUNCID:[0-9]+]] op1=7289175272376759421/>
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <COMBINED
; See if the call to func is registered, using the expected hotness type.
; op6=2 which is hotnessType::None.
; COMBINED-NEXT:    <COMBINED_PROFILE {{.*}} op9=[[FUNCID]] op10=2/>
; COMBINED-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() #0 !prof !2 {
entry:
    call void (...) @func()
    ret i32 0
}

declare void @func(...) #1

!2 = !{!"function_entry_count", i64 1}

; OLD: Index {{.*}} contains 1 nodes (1 functions, 0 alias, 0 globals) and 1 edges (0 refs and 1 calls)
; OLD-COMBINED: Index {{.*}} contains 2 nodes (2 functions, 0 alias, 0 globals) and 1 edges (0 refs and 1 calls)
