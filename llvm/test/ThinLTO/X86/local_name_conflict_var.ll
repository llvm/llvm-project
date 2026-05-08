; Test handling when multiple files with the same source file name contain
; static read only variables with the same name (which will have the same GUID
; in the combined index).

; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary -module-hash %s -o %t.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict_var1.ll -o %t1.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict_var2.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t4.bc %t.bc %t1.bc %t2.bc

; This module will import a() and b() which should cause the read only copy
; of baz from each of those modules to be imported. Check that the both are
; imported as local copies.
; RUN: llvm-lto -thinlto-action=import -exported-symbol=main %t.bc -thinlto-index=%t4.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORT
; IMPORT: @baz.llvm.{{.*}} = internal global i32 10
; IMPORT: @baz.llvm.{{.*}} = internal global i32 10

;; Now do the same but linking in a 3rd module, in which baz is a local function,
;; which has a non-call reference to it in c(). We should correctly mark all
;; summaries for baz as non-read only, which should cause the linkage type of
;; the imported copies of baz variables to be available_externally instead of
;; internal.
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict_var3.ll -o %t3.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t5.bc %t.bc %t1.bc %t2.bc %t3.bc
; RUN: llvm-lto -thinlto-action=import -exported-symbol=main %t.bc -thinlto-index=%t5.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORTNOREADONLY
; IMPORTNOREADONLY: @baz.llvm.{{.*}} = available_externally hidden global i32 10
; IMPORTNOREADONLY: @baz.llvm.{{.*}} = available_externally hidden global i32 10

;; Do this again but where the 3rd module calls local function baz.
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict_var3b.ll -o %t3b.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t5.bc %t.bc %t1.bc %t2.bc %t3b.bc
; RUN: llvm-lto -thinlto-action=import -exported-symbol=main %t.bc -thinlto-index=%t5.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORTNOREADONLY

;; Now link in the new module before the other 2 modules. We currently will stop
;; looking for a variable to import once we see the non-variable summary. This
;; is a compile time optimization. It is safe because we have marked all
;; summaries for baz as non-read-only, as shown above.
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict_var3.ll -o %t3.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t5.bc %t.bc %t3.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=import -exported-symbol=main %t.bc -thinlto-index=%t5.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=NOIMPORT
; NOIMPORT: @baz.llvm.{{.*}} = external hidden global i32, align 4
; NOIMPORT: @baz.llvm.{{.*}} = external hidden global i32, align 4

; ModuleID = 'local_name_conflict_var_main.o'
source_filename = "local_name_conflict_var_main.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define i32 @main() {
entry:
  %call1 = call i32 (...) @a()
  %call2 = call i32 (...) @b()
  %call3 = call i32 (...) @c()
  ret i32 0
}

declare i32 @a(...)
declare i32 @b(...)
declare i32 @c(...)
