; Test to check the callgraph in summary
; RUN: opt -write-relbf-to-summary -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s


; CHECK: <SOURCE_FILENAME
; CHECK-NEXT: <GLOBALVAR
; CHECK-NEXT: <FUNCTION
; "func"
; CHECK-NEXT: <FUNCTION op0=17 op1=4
; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; See if the call to func is registered.
; CHECK-NEXT:    <PERMODULE_RELBF {{.*}} op4=1 {{.*}} op7=256
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>
; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'undefinedglobmainfunc{{.*}}'


; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
    call void (...) @func()
    %u = load i32, i32* @undefinedglob
    ret i32 %u
}

declare void @func(...) #1
@undefinedglob = external global i32

; OLD: Index {{.*}} contains 1 nodes (1 functions, 0 alias, 0 globals) and 1 edges (0 refs and 1 calls)
