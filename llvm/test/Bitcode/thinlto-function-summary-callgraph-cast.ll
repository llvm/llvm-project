; Test to check the callgraph for calls to casts.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; PR34966

; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; "op7" is a call to "callee" function.
; CHECK-NEXT:    <PERMODULE {{.*}} op7=3 op8=[[ALIASID:[0-9]+]]/>
; "another_caller" has only references but no calls.
; CHECK-NEXT:    <PERMODULE {{.*}} op4=3 {{.*}} op7={{[0-9]+}}/>
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

define void @another_caller() {
    ; Test calls that aren't handled either as direct or indirect.
    call void select (i1 icmp eq (i32* @global, i32* null), void ()* @f, void ()* @g)()
    ret void
}

declare void @callee(...)

@analias = alias void (...), bitcast (void ()* @aliasee to void (...)*)

define void @aliasee() {
entry:
    ret void
}

declare void @f()
declare void @g()
@global = extern_weak global i32
