; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/merge-triple.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=optimize %t1.bc %t2.bc
; RUN: llvm-dis < %t1.bc.thinlto.imported.bc | FileCheck %s --check-prefix=CHECK1
; RUN: llvm-dis < %t2.bc.thinlto.imported.bc | FileCheck %s --check-prefix=CHECK2

target triple = "x86_64-apple-macosx10.12.0"

; CHECK1: target triple = "x86_64-apple-macosx10.12.0"
; CHECK2: target triple = "x86_64-apple-macosx10.11.0"
