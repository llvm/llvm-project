; Test to check the callgraph in summary
; RUN: opt -write-relbf-to-summary -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.o | llvm-as -write-relbf-to-summary -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

; CHECK: <SOURCE_FILENAME
; CHECK-NEXT: <GLOBALVAR
; CHECK-NEXT: <FUNCTION
; "func"
; CHECK-NEXT: <FUNCTION op0=17 op1=4
; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <FLAGS
; See if the call to func is registered.
; CHECK-NEXT:    <PERMODULE_RELBF {{.*}} op4=1 {{.*}} op9=256
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

; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (name: "func") ; guid = 7289175272376759421
; DIS: ^2 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 3, calls: ((callee: ^1, relbf: 256)), refs: (readonly ^3)))) ; guid = 15822663052811949562
; DIS: ^3 = gv: (name: "undefinedglob") ; guid = 18036901804029949403
