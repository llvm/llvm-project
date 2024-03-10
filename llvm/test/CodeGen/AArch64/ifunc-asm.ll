; RUN: llc -mtriple=arm64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=ELF
; RUN: llc -mtriple=arm64-apple-darwin %s -o - | FileCheck %s --check-prefixes=MACHO,ARM64
; RUN: llc -mtriple=arm64-apple-darwin %s -global-isel -o - | FileCheck %s --check-prefixes=MACHO,ARM64
; RUN: llc -mtriple=arm64e-apple-darwin %s -o - | FileCheck %s --check-prefixes=MACHO,PAUTH
; RUN: llc -mtriple=arm64e-apple-darwin %s -global-isel -o - | FileCheck %s --check-prefixes=MACHO,PAUTH

define internal ptr @the_resolver() {
entry:
  ret ptr null
}
; ELF:             .type the_resolver,@function
; ELF-NEXT:    the_resolver:

; MACHO:           .p2align 2
; MACHO-NEXT:  _the_resolver:


@global_ifunc = ifunc i32 (i32), ptr @the_resolver
; ELF:             .globl global_ifunc
; ELF-NEXT:        .type global_ifunc,@gnu_indirect_function
; ELF-NEXT:        .set global_ifunc, the_resolver

; MACHO:           .section __DATA,__data
; MACHO-NEXT:      .p2align 3, 0x0
; MACHO-NEXT:  _global_ifunc.lazy_pointer:
; ARM64-NEXT:      .quad _global_ifunc.stub_helper{{$}}
; PAUTH-NEXT:      .quad _global_ifunc.stub_helper@AUTH(ia,0)

; MACHO:           .section __TEXT,__text,regular,pure_instructions
; MACHO-NEXT:      .globl _global_ifunc
; MACHO-NEXT:      .p2align 2
; MACHO-NEXT:  _global_ifunc:
; MACHO-NEXT:      adrp    x16, _global_ifunc.lazy_pointer@GOTPAGE
; MACHO-NEXT:      ldr     x16, [x16, _global_ifunc.lazy_pointer@GOTPAGEOFF]
; MACHO-NEXT:      ldr     x16, [x16]
; ARM64-NEXT:      br      x16
; PAUTH-NEXT:      braaz   x16
; MACHO-NEXT:      .p2align        2
; MACHO-NEXT:  _global_ifunc.stub_helper:
; PAUTH-NEXT:      pacibsp
; MACHO-NEXT:      stp     x29, x30, [sp, #-16]!
; MACHO-NEXT:      mov     x29, sp
; MACHO-NEXT:      stp     x1, x0, [sp, #-16]!
; MACHO-NEXT:      stp     x3, x2, [sp, #-16]!
; MACHO-NEXT:      stp     x5, x4, [sp, #-16]!
; MACHO-NEXT:      stp     x7, x6, [sp, #-16]!
; MACHO-NEXT:      stp     d1, d0, [sp, #-16]!
; MACHO-NEXT:      stp     d3, d2, [sp, #-16]!
; MACHO-NEXT:      stp     d5, d4, [sp, #-16]!
; MACHO-NEXT:      stp     d7, d6, [sp, #-16]!
; MACHO-NEXT:      bl      _the_resolver
; MACHO-NEXT:      adrp    x16, _global_ifunc.lazy_pointer@GOTPAGE
; MACHO-NEXT:      ldr     x16, [x16, _global_ifunc.lazy_pointer@GOTPAGEOFF]
; MACHO-NEXT:      str     x0, [x16]
; MACHO-NEXT:      add     x16, x0, #0
; MACHO-NEXT:      ldp     d7, d6, [sp], #16
; MACHO-NEXT:      ldp     d5, d4, [sp], #16
; MACHO-NEXT:      ldp     d3, d2, [sp], #16
; MACHO-NEXT:      ldp     d1, d0, [sp], #16
; MACHO-NEXT:      ldp     x7, x6, [sp], #16
; MACHO-NEXT:      ldp     x5, x4, [sp], #16
; MACHO-NEXT:      ldp     x3, x2, [sp], #16
; MACHO-NEXT:      ldp     x1, x0, [sp], #16
; MACHO-NEXT:      ldp     x29, x30, [sp], #16
; PAUTH-NEXT:      autibsp
; PAUTH-NEXT:      eor     x17, x30, x30, lsl #1
; PAUTH-NEXT:      tbz     x17, #62, [[GOOD_SIG:Ltmp[0-9]+]]
; PAUTH-NEXT:      brk     #0xc471
; PAUTH-NEXT: [[GOOD_SIG]]:
; ARM64-NEXT:      br      x16
; PAUTH-NEXT:      braaz   x16


@weak_ifunc = weak ifunc i32 (i32), ptr @the_resolver
; ELF:             .type weak_ifunc,@gnu_indirect_function
; MACHO-NOT:       .weak_reference _weak_ifunc.lazy_pointer
; MACHO:       _weak_ifunc.lazy_pointer:
; MACHO:           .weak_reference _weak_ifunc
; MACHO:       _weak_ifunc:
; MACHO-NOT:       .weak_reference _weak_ifunc.stub_helper
; MACHO:       _weak_ifunc.stub_helper: