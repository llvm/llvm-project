; Test to check the callgraph for calls to casts.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; PR34966

; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <FLAGS
; "op7" is a call to "callee" function.
; CHECK-NEXT:    <PERMODULE_PROFILE {{.*}} op7=3 op8=0 op9=[[ALIASID:[0-9]+]]
; "another_caller" has only references but no calls.
; CHECK-NEXT:    <PERMODULE_PROFILE {{.*}}/>
; CHECK-NEXT:    <PERMODULE_PROFILE {{.*}} op0=[[ALIASEEID:[0-9]+]]
; CHECK-NEXT:    <ALIAS {{.*}} op0=[[ALIASID]] {{.*}} op2=[[ALIASEEID]]/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; ModuleID = 'thinlto-function-summary-callgraph-cast.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller() {
    call void @callee()
    call void @analias()
    ret void
}

define void @another_caller() {
    ; Test calls that aren't handled either as direct or indirect.
    call void getelementptr (i8, ptr @f, i64 ptrtoint (ptr @g to i64))()
    ret void
}

declare void @callee(...)

@analias = alias void (...), ptr @aliasee

define void @aliasee() {
entry:
    ret void
}

declare void @f()
declare void @g()
@global = extern_weak global i32
