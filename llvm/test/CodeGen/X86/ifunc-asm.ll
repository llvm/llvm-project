; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=ELF
; RUN: llc -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s --check-prefixes=MACHO

define internal ptr @foo_resolver() {
entry:
  ret i32 (i32)* null
}
; ELF:             .type foo_resolver,@function
; ELF-NEXT:    foo_resolver:

; MACHO:           .p2align 4, 0x90
; MACHO-NEXT:  _foo_resolver


@foo_ifunc = ifunc i32 (i32), ptr @foo_resolver
; ELF:             .globl foo_ifunc
; ELF-NEXT:        .type foo_ifunc,@gnu_indirect_function
; ELF-NEXT:        .set foo_ifunc, foo_resolver

; MACHO:           .section __DATA,__data
; MACHO-NEXT:      .p2align 3, 0x0
; MACHO-NEXT:  _foo_ifunc.lazy_pointer:
; MACHO-NEXT:      .quad _foo_ifunc.stub_helper
; MACHO-NEXT:      .section __TEXT,__text,regular,pure_instructions
; MACHO-NEXT:      .globl _foo_ifunc
; MACHO-NEXT:      .p2align 0, 0x90
; MACHO-NEXT:  _foo_ifunc:
; MACHO-NEXT:      jmpl   *_foo_ifunc.lazy_pointer(%rip)
; MACHO-NEXT:      .p2align 0, 0x90
; MACHO-NEXT:  _foo_ifunc.stub_helper:
; MACHO-NEXT:      pushq   %rax
; MACHO-NEXT:      pushq   %rdi
; MACHO-NEXT:      pushq   %rsi
; MACHO-NEXT:      pushq   %rdx
; MACHO-NEXT:      pushq   %rcx
; MACHO-NEXT:      pushq   %r8
; MACHO-NEXT:      pushq   %r9
; MACHO-NEXT:      callq   _foo_resolver
; MACHO-NEXT:      movq    %rax, _foo_ifunc.lazy_pointer(%rip)
; MACHO-NEXT:      popq    %r9
; MACHO-NEXT:      popq    %r8
; MACHO-NEXT:      popq    %rcx
; MACHO-NEXT:      popq    %rdx
; MACHO-NEXT:      popq    %rsi
; MACHO-NEXT:      popq    %rdi
; MACHO-NEXT:      popq    %rax
; MACHO-NEXT:      jmpl    *_foo_ifunc.lazy_pointer(%rip)

@weak_ifunc = weak ifunc i32 (i32), ptr @foo_resolver
; ELF:             .type weak_ifunc,@gnu_indirect_function
; MACHO-NOT:       .weak_reference _weak_ifunc.lazy_pointer
; MACHO:       _weak_ifunc.lazy_pointer:
; MACHO:           .weak_reference _weak_ifunc
; MACHO:       _weak_ifunc:
; MACHO-NOT:       .weak_reference _weak_ifunc.stub_helper
; MACHO:       _weak_ifunc.stub_helper:
