; Verify that auto-upgrading intrinsics works with Lazy loaded bitcode
; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %p/Inputs/autoupgrade.bc

; We can't use llvm-dis here, because it would do the autoupgrade itself.

; RUN: llvm-link  -summary-index=%t3.bc \
; RUN:            -import=globalfunc1:%p/Inputs/autoupgrade.bc %t.bc \
; RUN:     | llvm-bcanalyzer -dump | FileCheck %s

; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'mainglobalfunc1llvm.invariant.start.p0i8{{.*}}'

; Check that the summary is able to print the names despite the lack of
; string table in the legacy bitcode.
; RUN: llvm-dis %p/Inputs/autoupgrade.bc -o - \
; RUN:	   | FileCheck %s --check-prefix=SUMMARYNAMES
; SUMMARYNAMES: ^2 = gv: (name: "globalfunc2",
; SUMMARYNAMES: ^3 = gv: (name: "globalfunc1"

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @main() #0 {
entry:
  call void (...) @globalfunc1()
  ret i32 0
}

declare void @globalfunc1(...) #1
