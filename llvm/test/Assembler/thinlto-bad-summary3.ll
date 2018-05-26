; Test that we get appropriate error when parsing summary with unbalanced
; parentheses.
; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: error: found end of file while parsing summary entry

; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
    ret i32 1
}

^0 = module: (path: "{{.*}}thinlto-bad-summary1.ll", hash: (0, 0, 0, 0, 0))
; Missing a ')'
^1 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)) ; guid = 15822663052811949562
