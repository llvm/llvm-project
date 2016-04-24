;; RUN: opt -module-summary %s -o %t1.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t.index.bc %t1.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=REGULAR
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t.index.bc %t1.bc -o -  --exported-symbol=foo | llvm-dis -o - | FileCheck %s --check-prefix=INTERNALIZE

; REGULAR: define void @foo
; REGULAR: define void @bar
; INTERNALIZE: define void @foo
; INTERNALIZE: define internal void @bar

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @foo() {
    ret void
}
define void @bar() {
    ret void
}