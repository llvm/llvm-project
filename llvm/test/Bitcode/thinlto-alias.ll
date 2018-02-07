; Test to check the callgraph in summary
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: opt -module-summary %p/Inputs/thinlto-alias.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED

; CHECK: <SOURCE_FILENAME
; "main"
; CHECK-NEXT: <FUNCTION op0=0 op1=4
; "analias"
; CHECK-NEXT: <FUNCTION op0=4 op1=7
; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; See if the call to func is registered.
; The value id 1 matches the second FUNCTION record above.
; CHECK-NEXT:    <PERMODULE {{.*}} op5=1/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'mainanalias{{.*}}'

; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
; COMBINED-NEXT:    <FLAGS
; See if the call to analias is registered, using the expected value id.
; COMBINED-NEXT:    <VALUE_GUID op0=[[ALIASID:[0-9]+]] op1=-5751648690987223394/>
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID op0=[[ALIASEEID:[0-9]+]] op1=-1039159065113703048/>
; COMBINED-NEXT:    <COMBINED {{.*}} op6=[[ALIASID]]/>
; COMBINED-NEXT:    <COMBINED {{.*}}
; COMBINED-NEXT:    <COMBINED_ALIAS  {{.*}} op3=[[ALIASEEID]]
; COMBINED-NEXT:  </GLOBALVAL_SUMMARY_BLOCK

; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() {
entry:
    call void (...) @analias()
    ret i32 0
}

declare void @analias(...)
