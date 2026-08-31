; RUN: llc -mtriple=arm64e-apple-darwin %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64e-apple-darwin %s -global-isel -o - | FileCheck %s

attributes #0 = { "ptrauth-calls" "ptrauth-returns" }

define internal ptr @the_resolver() #0 {
entry:
  ret ptr null
}

; CHECK:           .p2align 2
; CHECK-NEXT:  _the_resolver:


@global_ifunc = ifunc i32 (i32), ptr @the_resolver
; CHECK:           .section __DATA,__data
; CHECK-NEXT:      .p2align 3, 0x0
; CHECK-NEXT:  _global_ifunc.lazy_pointer:
; CHECK-NEXT:      .quad _global_ifunc.stub_helper@AUTH(ia,0)

; CHECK:           .section __TEXT,__text,regular,pure_instructions
; CHECK-NEXT:      .globl _global_ifunc
; CHECK-NEXT:      .p2align 2
; CHECK-NEXT:  _global_ifunc:
; CHECK-NEXT:      adrp    x16, _global_ifunc.lazy_pointer@GOTPAGE
; CHECK-NEXT:      ldr     x16, [x16, _global_ifunc.lazy_pointer@GOTPAGEOFF]
; CHECK-NEXT:      ldr     x16, [x16]
; CHECK-NEXT:      braaz   x16
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  _global_ifunc.stub_helper:
; CHECK-NEXT:      pacibsp
; CHECK-NEXT:      stp     x29, x30, [sp, #-16]!
; CHECK-NEXT:      mov     x29, sp
; CHECK-NEXT:      stp     x1, x0, [sp, #-16]!
; CHECK-NEXT:      stp     x3, x2, [sp, #-16]!
; CHECK-NEXT:      stp     x5, x4, [sp, #-16]!
; CHECK-NEXT:      stp     x7, x6, [sp, #-16]!
; CHECK-NEXT:      stp     d1, d0, [sp, #-16]!
; CHECK-NEXT:      stp     d3, d2, [sp, #-16]!
; CHECK-NEXT:      stp     d5, d4, [sp, #-16]!
; CHECK-NEXT:      stp     d7, d6, [sp, #-16]!
; CHECK-NEXT:      bl      _the_resolver
; CHECK-NEXT:      adrp    x16, _global_ifunc.lazy_pointer@GOTPAGE
; CHECK-NEXT:      ldr     x16, [x16, _global_ifunc.lazy_pointer@GOTPAGEOFF]
; CHECK-NEXT:      str     x0, [x16]
; CHECK-NEXT:      add     x16, x0, #0
; CHECK-NEXT:      ldp     d7, d6, [sp], #16
; CHECK-NEXT:      ldp     d5, d4, [sp], #16
; CHECK-NEXT:      ldp     d3, d2, [sp], #16
; CHECK-NEXT:      ldp     d1, d0, [sp], #16
; CHECK-NEXT:      ldp     x7, x6, [sp], #16
; CHECK-NEXT:      ldp     x5, x4, [sp], #16
; CHECK-NEXT:      ldp     x3, x2, [sp], #16
; CHECK-NEXT:      ldp     x1, x0, [sp], #16
; CHECK-NEXT:      ldp     x29, x30, [sp], #16
; CHECK-NEXT:      braaz   x16


@weak_ifunc = weak ifunc i32 (i32), ptr @the_resolver
; CHECK-NOT:       .weak_reference _weak_ifunc.lazy_pointer
; CHECK:       _weak_ifunc.lazy_pointer:
; CHECK:           .weak_reference _weak_ifunc{{$}}
; CHECK:       _weak_ifunc:
; CHECK-NOT:       .weak_reference _weak_ifunc.stub_helper
; CHECK:       _weak_ifunc.stub_helper: