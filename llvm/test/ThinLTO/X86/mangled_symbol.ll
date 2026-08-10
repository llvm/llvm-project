; RUN: opt -module-summary %s -o %t1.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc
;; Check baseline when both of them internalized when not exported.
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t.index.bc %t1.bc -o - --exported-symbol=_exported | llvm-dis -o - | FileCheck %s --check-prefix=INTERNALIZED
;; Check symbols are exported, including the ones with `\01` prefix.
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t.index.bc %t1.bc -o - --exported-symbol=_exported --exported-symbol=_extern_not_mangled --exported-symbol=_extern_mangled | llvm-dis -o - | FileCheck %s --check-prefix=EXPORTED

; INTERNALIZED: define internal void @extern_not_mangled
; INTERNALIZED: define internal void @"\01_extern_mangled"
; EXPORTED: define void @extern_not_mangled
; EXPORTED: define void @"\01_extern_mangled"

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @exported() {
  ret void
}

define void @extern_not_mangled() {
  ret void
}

define void @"\01_extern_mangled"() {
  ret void
}
