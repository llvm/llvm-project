; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/section.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; Check that we don't promote 'var_with_section'
; RUN: llvm-lto -thinlto-action=promote %t2.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=PROMOTE
; PROMOTE: @var_with_section = internal global i32 0, section "some_section"

; RUN: llvm-lto -thinlto-action=import %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORT
; Check that section prevent import of @reference_gv_with_section.
; IMPORT: declare void @reference_gv_with_section()
; Canary to check that importing is correctly set up.
; IMPORT: define available_externally void @foo()
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"


define i32 @main() {
    call void @reference_gv_with_section()
    call void @foo()
    ret i32 42
}


declare void @reference_gv_with_section()
declare void @foo()
