; Test to check the callgraph in summary
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: opt -module-summary %p/Inputs/thinlto-function-summary-callgraph.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED

; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; See if the call to func is registered, using the expected callsite count
; and value id matching the subsequent value symbol table.
; CHECK-NEXT:    <PERMODULE {{.*}} op4=[[FUNCID:[0-9]+]] op5=1/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>
; CHECK-NEXT:  <VALUE_SYMTAB
; CHECK-NEXT:    <FNENTRY {{.*}} record string = 'main'
; External function func should have entry with value id FUNCID
; CHECK-NEXT:    <ENTRY {{.*}} op0=[[FUNCID]] {{.*}} record string = 'func'
; CHECK-NEXT:  </VALUE_SYMTAB>

; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
; COMBINED-NEXT:    <COMBINED
; See if the call to func is registered, using the expected callsite count
; and value id matching the subsequent value symbol table.
; COMBINED-NEXT:    <COMBINED {{.*}} op5=[[FUNCID:[0-9]+]] op6=1/>
; COMBINED-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>
; COMBINED-NEXT:  <VALUE_SYMTAB
; Entry for function func should have entry with value id FUNCID
; COMBINED-NEXT:    <COMBINED_ENTRY {{.*}} op0=[[FUNCID]] op1=7289175272376759421/>
; COMBINED-NEXT:    <COMBINED
; COMBINED-NEXT:  </VALUE_SYMTAB>

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
