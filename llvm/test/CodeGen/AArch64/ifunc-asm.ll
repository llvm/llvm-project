; RUN: llc -mtriple=arm64-unknown-linux-gnu %s -filetype=asm -o - | FileCheck %s --check-prefixes=ELF
; RUN: llc -mtriple=arm64-apple-darwin %s -filetype=asm -o - -arm64-darwin-ifunc-symbol_resolver=always | FileCheck %s --check-prefixes=MACHO,MACHO-LINKER
; RUN: llc -mtriple=arm64-apple-darwin %s -filetype=asm -o - -arm64-darwin-ifunc-symbol_resolver=if_supported | FileCheck %s --check-prefixes=MACHO,MACHO-DEFAULT
; RUN: llc -mtriple=arm64-apple-darwin %s -filetype=asm -o - -arm64-darwin-ifunc-symbol_resolver=never | FileCheck %s --check-prefixes=MACHO,MACHO-MANUAL
; RUN: llc -mtriple=arm64-apple-darwin %s -filetype=asm -o - | FileCheck %s --check-prefixes=MACHO,MACHO-MANUAL

define internal ptr @foo_resolver() {
entry:
  ret ptr null
}
; ELF:                    .type foo_resolver,@function
; ELF-NEXT:           foo_resolver:

; MACHO:                  .p2align 2
; MACHO-NEXT:         _foo_resolver


@foo_ifunc = ifunc i32 (i32), ptr @foo_resolver
; ELF:                    .globl foo_ifunc
; ELF-NEXT:               .type foo_ifunc,@gnu_indirect_function
; ELF-NEXT:               .set foo_ifunc, foo_resolver

; MACHO-LINKER:           .globl _foo_ifunc
; MACHO-LINKER-NEXT:      .p2align 2
; MACHO-LINKER-NEXT:  _foo_ifunc:
; MACHO-LINKER-NEXT:      .symbol_resolver _foo_ifunc
; MACHO-LINKER-NEXT:      b _foo_resolver

; MACHO-DEFAULT:          .globl _foo_ifunc
; MACHO-DEFAULT-NEXT:     .p2align 2
; MACHO-DEFAULT-NEXT: _foo_ifunc:
; MACHO-DEFAULT-NEXT:     .symbol_resolver _foo_ifunc
; MACHO-DEFAULT-NEXT:     b _foo_resolver

; MACHO-MANUAL:           .section __DATA,__data
; MACHO-MANUAL-NEXT:      .globl _foo_ifunc.lazy_pointer
; MACHO-MANUAL-NEXT:  _foo_ifunc.lazy_pointer:
; MACHO-MANUAL-NEXT:      .quad _foo_ifunc.stub_helper

; MACHO-MANUAL:           .section __TEXT,__text,regular,pure_instructions
; MACHO-MANUAL-NEXT:      .globl _foo_ifunc
; MACHO-MANUAL-NEXT:      .p2align 2
; MACHO-MANUAL-NEXT:  _foo_ifunc:
; MACHO-MANUAL-NEXT:      adrp    x16, _foo_ifunc.lazy_pointer@GOTPAGE
; MACHO-MANUAL-NEXT:      ldr     x16, [x16, _foo_ifunc.lazy_pointer@GOTPAGEOFF]
; MACHO-MANUAL-NEXT:      ldr     x16, [x16]
; MACHO-MANUAL-NEXT:      br      x16
; MACHO-MANUAL-NEXT:      .globl  _foo_ifunc.stub_helper
; MACHO-MANUAL-NEXT:      .p2align        2
; MACHO-MANUAL-NEXT:  _foo_ifunc.stub_helper:
; MACHO-MANUAL-NEXT:      stp     x29, x30, [sp, #-16]!
; MACHO-MANUAL-NEXT:      mov     x29, sp
; MACHO-MANUAL-NEXT:      stp     x1, x0, [sp, #-16]!
; MACHO-MANUAL-NEXT:      stp     x3, x2, [sp, #-16]!
; MACHO-MANUAL-NEXT:      stp     x5, x4, [sp, #-16]!
; MACHO-MANUAL-NEXT:      stp     x7, x6, [sp, #-16]!
; MACHO-MANUAL-NEXT:      stp     d1, d0, [sp, #-16]!
; MACHO-MANUAL-NEXT:      stp     d3, d2, [sp, #-16]!
; MACHO-MANUAL-NEXT:      stp     d5, d4, [sp, #-16]!
; MACHO-MANUAL-NEXT:      stp     d7, d6, [sp, #-16]!
; MACHO-MANUAL-NEXT:      bl      _foo_resolver
; MACHO-MANUAL-NEXT:      adrp    x16, _foo_ifunc.lazy_pointer@GOTPAGE
; MACHO-MANUAL-NEXT:      ldr     x16, [x16, _foo_ifunc.lazy_pointer@GOTPAGEOFF]
; MACHO-MANUAL-NEXT:      str     x0, [x16]
; MACHO-MANUAL-NEXT:      add     x16, x0, #0
; MACHO-MANUAL-NEXT:      ldp     d7, d6, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     d5, d4, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     d3, d2, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     d1, d0, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     x7, x6, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     x5, x4, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     x3, x2, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     x1, x0, [sp], #16
; MACHO-MANUAL-NEXT:      ldp     x29, x30, [sp], #16
; MACHO-MANUAL-NEXT:      br      x16


@weak_ifunc = weak ifunc i32 (i32), ptr @foo_resolver
; ELF:             .type weak_ifunc,@gnu_indirect_function
; MACHO-LINKER:    .symbol_resolver _weak_ifunc
; MACHO-MANUAL:    _weak_ifunc.stub_helper:
; MACHO-DEFEAULT:  _weak_ifunc.stub_helper: