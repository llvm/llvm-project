; Test to ensure we don't internalize or treat as dead a global value
; with a valid C identifier section name. Otherwise, ELF linker generation of
; __start_"sectionname" and __stop_"sectionname" symbols would not occur and
; we can end up with undefined references at link time.

; First try RegularLTO
; RUN: opt %s -o %t.o
; RUN: llvm-lto2 dump-symtab %t.o | FileCheck %s --check-prefix=SYMTAB
; RUN: opt %p/Inputs/global_with_section.ll -o %t2.o
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=save-temps \
; RUN:     -o %t3.o %t.o %t2.o
; Check results of internalization
; RUN: llvm-dis %t3.o.0.2.internalize.bc -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK2-REGULARLTO

; Next try ThinLTO
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-lto2 dump-symtab %t.o | FileCheck %s --check-prefix=SYMTAB
; RUN: opt -module-summary %p/Inputs/global_with_section.ll -o %t2.o
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=save-temps \
; RUN:     -o %t3.o %t.o %t2.o
; Check results of internalization
; RUN: llvm-dis %t.o.2.internalize.bc -o - | FileCheck %s
; RUN: llvm-dis %t2.o.2.internalize.bc -o - | FileCheck %s --check-prefix=CHECK2-THINLTO

; SYMTAB: deadfunc_with_section
; SYMTAB-NEXT: section some_other_section
; SYMTAB-NEXT: deadfunc_with_nonC_section
; SYMTAB-NEXT: section .nonCsection
; SYMTAB-NEXT: deadfunc2_called_from_section
; SYMTAB-NEXT: deadfunc2_called_from_nonC_section
; SYMTAB-NEXT: var_with_section
; SYMTAB-NEXT: section some_section
; SYMTAB-NEXT: var_with_nonC_section
; SYMTAB-NEXT: section .nonCsection

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; We should not internalize @var_with_section due to section
; CHECK-DAG: @var_with_section = dso_local global i32 0, section "some_section"
@var_with_section = global i32 0, section "some_section"

; Confirm via a variable with a non-C identifier section that we are getting
; the expected internalization.
; CHECK-DAG: @var_with_nonC_section = internal dso_local global i32 0, section ".nonCsection"
@var_with_nonC_section = global i32 0, section ".nonCsection"

; We should not internalize @deadfunc_with_section due to section
; CHECK-DAG: define dso_local void @deadfunc_with_section() section "some_other_section"
define void @deadfunc_with_section() section "some_other_section" {
  call void @deadfunc2_called_from_section()
  ret void
}

; Confirm via a function with a non-C identifier section that we are getting
; the expected internalization.
; CHECK-DAG: define internal dso_local void @deadfunc_with_nonC_section() section ".nonCsection"
define void @deadfunc_with_nonC_section() section ".nonCsection" {
  call void @deadfunc2_called_from_nonC_section()
  ret void
}

; In RegularLTO mode, where we have combined all the IR,
; @deadfunc2_called_from_section can be internalized.
; CHECK2-REGULARLTO: define internal dso_local void @deadfunc2_called_from_section
; In ThinLTO mode, we can't internalize it as it needs to be preserved
; (due to the access from @deadfunc_with_section which must be preserved), and
; can't be internalized since the reference is from a different module.
; CHECK2-THINLTO: define dso_local void @deadfunc2_called_from_section
declare void @deadfunc2_called_from_section()

; Confirm when called from a function with a non-C identifier section that we
; are getting the expected internalization.
; CHECK2-REGULARLTO: define internal dso_local void @deadfunc2_called_from_nonC_section
; CHECK2-THINLTO: define internal dso_local void @deadfunc2_called_from_nonC_section
declare void @deadfunc2_called_from_nonC_section()
