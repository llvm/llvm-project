; Test to check the callgraph for call to alias in module.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <FLAGS
; CHECK-NEXT:    <PERMODULE {{.*}} op4=0 op5=0 op6=0 op7=[[ALIASID:[0-9]+]]/>
; CHECK-NEXT:    <PERMODULE {{.*}} op0=[[ALIASEEID:[0-9]+]]
; CHECK-NEXT:    <ALIAS {{.*}} op0=[[ALIASID]] {{.*}} op2=[[ALIASEEID]]/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; ModuleID = 'thinlto-alias2.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
    call void (...) @analias()
    ret i32 0
}

@analias = alias void (...), bitcast (void ()* @aliasee to void (...)*)

define void @aliasee() #0 {
entry:
    ret void
}

