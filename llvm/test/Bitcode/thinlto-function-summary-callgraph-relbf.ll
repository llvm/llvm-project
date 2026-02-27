;; Test to ensure that we can still read old bitcode containing the now
;; deprected PERMODULE_RELBF record. It should be read and the relbf itself
;; ignored.

; RUN: llvm-bcanalyzer -dump %S/Inputs/thinlto-function-summary-callgraph-relbf.bc | FileCheck %s
; RUN: llvm-dis -o - %S/Inputs/thinlto-function-summary-callgraph-relbf.bc | FileCheck %s --check-prefix=DIS

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

; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (name: "func") ; guid = 7289175272376759421
;; We should have ignored the relbf in the old bitcode, and it no longer shows
;; up in the summary.
; DIS: ^2 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0, importType: definition), insts: 3, calls: ((callee: ^1)), refs: (readonly ^3)))) ; guid = 15822663052811949562
; DIS: ^3 = gv: (name: "undefinedglob") ; guid = 18036901804029949403
