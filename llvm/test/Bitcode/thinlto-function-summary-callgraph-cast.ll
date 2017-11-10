; Test to check the callgraph for calls to casts.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; PR34966

; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; "op7=1" is a call to "callee" function.
; CHECK-NEXT:    <PERMODULE {{.*}} op7=1 op8=[[ALIASID:[0-9]+]]/>
; CHECK-NEXT:    <PERMODULE {{.*}} op0=[[ALIASEEID:[0-9]+]]
; CHECK-NEXT:    <ALIAS {{.*}} op0=[[ALIASID]] {{.*}} op2=[[ALIASEEID]]/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; ModuleID = 'thinlto-function-summary-callgraph-cast.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller() {
    call void bitcast (void (...)* @callee to void ()*)()
    call void bitcast (void (...)* @analias to void ()*)()
    ret void
}

declare void @callee(...)

@analias = alias void (...), bitcast (void ()* @aliasee to void (...)*)

define void @aliasee() {
entry:
    ret void
}
