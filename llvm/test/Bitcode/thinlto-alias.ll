; Test to check the callgraph in summary
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: opt -module-summary %p/Inputs/thinlto-alias.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED

; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; See if the call to func is registered, using the expected callsite count
; and value id matching the subsequent value symbol table.
; CHECK-NEXT:    <PERMODULE {{.*}} op4=[[FUNCID:[0-9]+]]/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>
; CHECK-NEXT:  <VALUE_SYMTAB
; CHECK-NEXT:    <FNENTRY {{.*}} record string = 'main'
; External function analias should have entry with value id FUNCID
; CHECK-NEXT:    <ENTRY {{.*}} op0=[[FUNCID]] {{.*}} record string = 'analias'
; CHECK-NEXT:  </VALUE_SYMTAB>

; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
; See if the call to analias is registered, using the expected callsite count
; and value id matching the subsequent value symbol table.
; COMBINED-NEXT:    <COMBINED {{.*}} op5=[[ALIASID:[0-9]+]]/>
; Followed by the alias and aliasee
; COMBINED-NEXT:    <COMBINED {{.*}}
; COMBINED-NEXT:    <COMBINED_ALIAS  {{.*}} op3=[[ALIASEEID:[0-9]+]]
; COMBINED-NEXT:  </GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:  <VALUE_SYMTAB
; Entry for function func should have entry with value id ALIASID
; COMBINED-NEXT:    <COMBINED_ENTRY {{.*}} op0=[[ALIASID]] op1=-5751648690987223394/>
; COMBINED-NEXT:    <COMBINED
; COMBINED-NEXT:    <COMBINED_ENTRY {{.*}} op0=[[ALIASEEID]] op1=-1039159065113703048/>
; COMBINED-NEXT:  </VALUE_SYMTAB>

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
