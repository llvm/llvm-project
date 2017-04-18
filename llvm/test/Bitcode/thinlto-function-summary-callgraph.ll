; Test to check the callgraph in summary
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; RUN: opt -module-summary %p/Inputs/thinlto-function-summary-callgraph.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED

; Check parsing for old summary versions generated from this file.
; RUN: llvm-lto -thinlto-index-stats %p/Inputs/thinlto-function-summary-callgraph.1.bc  | FileCheck %s --check-prefix=OLD
; RUN: llvm-lto -thinlto-index-stats %p/Inputs/thinlto-function-summary-callgraph-combined.1.bc  | FileCheck %s --check-prefix=OLD-COMBINED

; CHECK: <SOURCE_FILENAME
; CHECK-NEXT: <FUNCTION
; "func"
; CHECK-NEXT: <FUNCTION op0=4 op1=4
; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; See if the call to func is registered.
; CHECK-NEXT:    <PERMODULE {{.*}} op4=1/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>
; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'mainfunc'


; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
; COMBINED-NEXT:    <VALUE_GUID op0=[[FUNCID:[0-9]+]] op1=7289175272376759421/>
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <COMBINED
; See if the call to func is registered.
; COMBINED-NEXT:    <COMBINED {{.*}} op5=[[FUNCID]]/>
; COMBINED-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
    call void (...) @func()
    ret i32 0
}

declare void @func(...) #1

; OLD: Index {{.*}} contains 1 nodes (1 functions, 0 alias, 0 globals) and 1 edges (0 refs and 1 calls)
; OLD-COMBINED: Index {{.*}} contains 2 nodes (2 functions, 0 alias, 0 globals) and 1 edges (0 refs and 1 calls)
