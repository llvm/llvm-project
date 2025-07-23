; RUN: llc < %s | FileCheck %s
; RUN: llc -function-sections -data-sections < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: .section .text.implicit_section_func,"ax",@llvm_cfi_jump_table,2
define void @implicit_section_func() !elf_section_properties !{i32 1879002126, i32 2} {
  ret void
}
; CHECK: .section foo,"ax",@llvm_cfi_jump_table,4
define void @explicit_section_func() section "foo" !elf_section_properties !{i32 1879002126, i32 4} {
  ret void
}

; CHECK: .section .data.implicit_section_global,"aw",@llvm_cfi_jump_table,8
@implicit_section_global = global i32 1, !elf_section_properties !{i32 1879002126, i32 8}
; CHECK: .section bar,"aw",@llvm_cfi_jump_table,16
@explicit_section_global = global i32 1, !elf_section_properties !{i32 1879002126, i32 16}, section "bar"
