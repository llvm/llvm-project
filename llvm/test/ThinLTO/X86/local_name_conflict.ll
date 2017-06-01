; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary -module-hash %s -o %t.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict1.ll -o %t2.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict2.ll -o %t3.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t4.bc %t.bc %t2.bc %t3.bc

; Make sure foo is promoted and renamed without complaint in both
; Inputs/local_name_conflict1.ll and Inputs/local_name_conflict2.ll
; FIXME: Once the importer is fixed to import the correct copy of the
; local, we should be able to verify that via an import action.
; RUN: llvm-lto -thinlto-action=promote %t2.bc -thinlto-index=%t4.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=EXPORTSTATIC
; RUN: llvm-lto -thinlto-action=promote %t3.bc -thinlto-index=%t4.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=EXPORTSTATIC
; EXPORTSTATIC: define hidden i32 @foo.llvm.

; ModuleID = 'local_name_conflict_main.o'
source_filename = "local_name_conflict_main.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call = call i32 (...) @b()
  ret i32 %call
}

declare i32 @b(...)
