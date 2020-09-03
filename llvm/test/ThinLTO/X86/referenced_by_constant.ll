; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/referenced_by_constant.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; Check the import side: we currently import bar() and also promote/import
; referenced constant objects. There is stll a room for improvement: we
; can make a local copy of someglobal and someglobal2 because they are both
; 'unnamed_addr' constants. This should eventually be done as well.
; RUN: llvm-lto -thinlto-action=import -import-constants-with-refs %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o -   | FileCheck %s --check-prefix=IMPORT
; IMPORT: @someglobal.llvm.0 = available_externally hidden unnamed_addr constant i8* bitcast (void ()* @referencedbyglobal to i8*)
; IMPORT: @someglobal2.llvm.0 = available_externally hidden unnamed_addr constant i8* bitcast (void ()* @localreferencedbyglobal.llvm.0 to i8*)
; IMPORT: define available_externally void @bar()

; Check the export side: we currently only export bar(), which causes
; @someglobal and @someglobal2 to be promoted (see above).
; RUN: llvm-lto -thinlto-action=promote -import-constants-with-refs %t2.bc -thinlto-index=%t3.bc -o - | llvm-dis -o -   | FileCheck %s --check-prefix=EXPORT
; EXPORT: @someglobal.llvm.0 = hidden unnamed_addr constant
; EXPORT: @someglobal2.llvm.0 = hidden unnamed_addr constant
; EXPORT: define void @referencedbyglobal()
; EXPORT: define hidden void @localreferencedbyglobal.llvm

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare void @bar()

define void @foo() {
    call void @bar()
    ret void
}
