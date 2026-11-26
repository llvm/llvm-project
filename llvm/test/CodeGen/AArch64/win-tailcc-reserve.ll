; RUN: llc -mtriple=aarch64-windows %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-windows -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND

; Tail calls that change the stack argument size, on Windows. Stack arguments
; for the tail call go in the area reserved at the top of the frame, and the
; epilogue pops by a different amount from the prologue's allocation, so it
; must have its own unwind codes.

declare swifttailcc void @callee12(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare swifttailcc void @callee1(i64)
declare swifttailcc void @callee9(i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare swifttailcc void @callee11(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare void @use(ptr)

; The tail callee needs 32 bytes of stack arguments but this function has none.
; CHECK-LABEL: grow:
; CHECK:         str x30, [sp, #-48]!
; CHECK-NEXT:    .seh_save_reg_x x30, 48
; CHECK:         .seh_endprologue
; CHECK:         bl use
; CHECK-DAG:     stp [[A:x[0-9]+]], [[A]], [sp, #32]
; CHECK-DAG:     stp [[A]], [[A]], [sp, #16]
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    ldr x30, [sp]
; CHECK-NEXT:    .seh_save_reg x30, 0
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @grow(i64 %a) {
  call void @use(ptr null)
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; This function receives stack arguments the tail callee doesn't take.
; CHECK-LABEL: shrink:
; CHECK:         str x19, [sp, #-16]!
; CHECK-NEXT:    .seh_save_reg_x x19, 16
; CHECK:         .seh_endprologue
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    ldr x19, [sp], #16
; CHECK-NEXT:    .seh_save_reg_x x19, 16
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee1
define swifttailcc void @shrink(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
  call void asm sideeffect "", "~{x19}"()
  musttail call swifttailcc void @callee1(i64 %a)
  ret void
}

; With a frame pointer the epilogue restores SP from it, and the tail call
; arguments are addressed relative to it.
; CHECK-LABEL: grow_dynamic:
; CHECK:         stp x29, x30, [sp, #-48]!
; CHECK-NEXT:    .seh_save_fplr_x 48
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    .seh_set_fp
; CHECK:         .seh_endprologue
; CHECK:         bl use
; CHECK-DAG:     stp [[A:x[0-9]+]], [[A]], [x29, #32]
; CHECK-DAG:     stp [[A]], [[A]], [x29, #16]
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    mov sp, x29
; CHECK-NEXT:    .seh_set_fp
; CHECK-NEXT:    ldp x29, x30, [sp]
; CHECK-NEXT:    .seh_save_fplr 0
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @grow_dynamic(i64 %a, i64 %n) {
  %buf = alloca i64, i64 %n
  call void @use(ptr %buf)
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; Saved registers are above the locals but below the reserved area, so the
; tail call arguments don't overwrite them.
; CHECK-LABEL: grow_csr:
; CHECK:         str x19, [sp, #-128]!
; CHECK-NEXT:    .seh_save_reg_x x19, 128
; CHECK:         .seh_endprologue
; CHECK-DAG:     stp [[A:x[0-9]+]], [[A]], [sp, #112]
; CHECK-DAG:     stp [[A]], [[A]], [sp, #96]
; CHECK:         .seh_startepilogue
; CHECK:         ldr x19, [sp]
; CHECK-NEXT:    .seh_save_reg x19, 0
; CHECK-NEXT:    add sp, sp, #96
; CHECK-NEXT:    .seh_stackalloc 96
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @grow_csr(i64 %a) {
  call void asm sideeffect "", "~{x19},~{x21},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{d8},~{d9},~{d10},~{d11}"()
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; The unwind info must describe the epilogue with its own codes: it pops less
; than the prologue allocated, because the reserved area holds the arguments for
; the tail call.
; UNWIND-LABEL: Function: grow (
; UNWIND:       Prologue [
; UNWIND-NEXT:    str x30, [sp, #-48]!
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND:       Opcodes [
; UNWIND-NEXT:    ldr x30, [sp, #0]
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: shrink (
; UNWIND:       Epilogue [
; UNWIND-NEXT:    ldr x19, [sp], #16
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: grow_dynamic (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    mov sp, fp
; UNWIND-NEXT:    ldp x29, x30, [sp, #0]
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: grow_csr (
; UNWIND:       Opcodes [
; UNWIND:         ldr x19, [sp, #0]
; UNWIND-NEXT:    add sp, #96
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]

; Both this function and its tail callee have stack arguments, and the callee
; has fewer. The epilogue pops the difference, then the callee's arguments are
; left for the callee to pop, which the unwinder must do after end_c.
; CHECK-LABEL: shrink_both:
; CHECK:         .seh_endprologue
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    ldr x19, [sp], #16
; CHECK-NEXT:    .seh_save_reg_x x19, 16
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee9
define swifttailcc void @shrink_both(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
  call void asm sideeffect "", "~{x19}"()
  musttail call swifttailcc void @callee9(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; The callee has more stack arguments than this function, which has some.
; CHECK-LABEL: grow_both:
; CHECK:         .seh_endprologue
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    ldr x19, [sp]
; CHECK-NEXT:    .seh_save_reg x19, 0
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee11
define swifttailcc void @grow_both(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
  call void asm sideeffect "", "~{x19}"()
  musttail call swifttailcc void @callee11(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; The callee has the same stack argument size, so there is nothing to pop in
; the epilogue, but the unwinder still has to pop the callee's arguments.
; CHECK-LABEL: same_both:
; CHECK:         .seh_endprologue
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    ldr x19, [sp], #16
; CHECK-NEXT:    .seh_save_reg_x x19, 16
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee9
define swifttailcc void @same_both(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
  call void asm sideeffect "", "~{x19}"()
  musttail call swifttailcc void @callee9(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; UNWIND-LABEL: Function: shrink_both (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    ldr x19, [sp], #16
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: grow_both (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    ldr x19, [sp, #0]
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: same_both (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    ldr x19, [sp], #16
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
