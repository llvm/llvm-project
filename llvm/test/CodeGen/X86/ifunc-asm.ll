; RUN: llc -filetype=asm -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=ELF
; RUN: llc -filetype=asm -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s --check-prefixes=MACHO

define internal ptr @foo_resolver() {
entry:
  ret ptr null
}
; ELF: .type foo_resolver,@function
; ELF-NEXT: foo_resolver:

; MACHO: .p2align 4, 0x90
; MACHO-NEXT: _foo_resolver


@foo_ifunc = ifunc i32 (i32), ptr @foo_resolver
; ELF:      .globl foo_ifunc
; ELF-NEXT: .type foo_ifunc,@gnu_indirect_function
; ELF-NEXT: .set foo_ifunc, foo_resolver

; MACHO:      .globl _foo_ifunc
; MACHO-NEXT: .p2align 4, 0x90
; MACHO-NEXT: _foo_ifunc:
; MACHO-NEXT: .symbol_resolver _foo_ifunc
; MACHO-NEXT: jmp _foo_resolver
