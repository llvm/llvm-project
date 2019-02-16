; Test handling when two files with the same source file name contain
; static functions/variables with the same name (which will have the same GUID
; in the combined index.

; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary -module-hash %s -o %t.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict1.ll -o %t2.bc
; RUN: opt -module-summary -module-hash %p/Inputs/local_name_conflict2.ll -o %t3.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t4.bc %t.bc %t2.bc %t3.bc

; This module will import b() which should cause the copy of foo and baz from
; that module (%t3.bc) to be imported. Check that the imported reference's
; promoted name matches the imported copy.
; RUN: llvm-lto -thinlto-action=import %t.bc -thinlto-index=%t4.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORT
; IMPORT: @baz.llvm.[[HASH:[0-9]+]] = available_externally hidden constant i32 10, align 4
; IMPORT: call i32 @foo.llvm.[[HASH]]
; IMPORT: define available_externally hidden i32 @foo.llvm.[[HASH]]()

; The copy in %t2.bc should not be exported/promoted/renamed
; RUN: llvm-lto -thinlto-action=promote %t2.bc -thinlto-index=%t4.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=NOEXPORTSTATIC
; NOEXPORTSTATIC: @baz = internal constant i32 10, align 4
; NOEXPORTSTATIC: define internal i32 @foo()

; Make sure foo is promoted and renamed without complaint in %t3.bc.
; RUN: llvm-lto -thinlto-action=promote %t3.bc -thinlto-index=%t4.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=EXPORTSTATIC
; EXPORTSTATIC: @baz.llvm.{{.*}} = hidden constant i32 10, align 4
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
