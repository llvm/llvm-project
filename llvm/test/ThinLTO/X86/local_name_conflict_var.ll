; Test handling when two files with the same source file name contain
; static read only variables with the same name (which will have the same GUID
; in the combined index).

; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary -module-hash %s -o %t.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict_var1.ll -o %t2.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict_var2.ll -o %t3.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t4.bc %t.bc %t2.bc %t3.bc

; This module will import a() and b() which should cause the read only copy
; of baz from each of those modules to be imported. Check that the both are
; imported as local copies.
; RUN: llvm-lto -thinlto-action=import -exported-symbol=main %t.bc -thinlto-index=%t4.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORT
; IMPORT: @baz.llvm.{{.*}} = internal global i32 10
; IMPORT: @baz.llvm.{{.*}} = internal global i32 10

; ModuleID = 'local_name_conflict_var_main.o'
source_filename = "local_name_conflict_var_main.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define i32 @main() {
entry:
  %call1 = call i32 (...) @a()
  %call2 = call i32 (...) @b()
  ret i32 0
}

declare i32 @a(...)
declare i32 @b(...)
