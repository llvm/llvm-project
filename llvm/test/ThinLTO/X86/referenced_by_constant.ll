; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/referenced_by_constant.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; Check the import side: we import bar() and @someglobal, but not @referencedbyglobal()
; RUN: llvm-lto -thinlto-action=import %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o -   | FileCheck %s --check-prefix=IMPORT
; IMPORT: @someglobal.llvm.0 =
; IMPORT: define available_externally void @bar()
; IMPORT: declare void @referencedbyglobal()

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare void @bar()

define void @foo() {
    call void @bar()
    ret void
}

