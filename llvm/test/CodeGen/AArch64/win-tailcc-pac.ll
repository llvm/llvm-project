; RUN: llc -mtriple=aarch64-windows -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,HINT
; RUN: llc -mtriple=aarch64-windows -mattr=+pauth -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,PAUTH
; RUN: llc -mtriple=aarch64-windows -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND

; When a tail call, or a return from a function with callee-popped stack
; arguments, changes the stack argument size, the epilogue doesn't leave SP at
; its value on entry. The Windows unwinder applies the pac_sign_lr unwind code
; using whatever SP it has at that point, which wouldn't be the SP that the
; return address was signed with, so the return address is authenticated before
; the epilogue starts, where the prologue's unwind codes still apply, and the
; authenticated return address stays in LR rather than being restored by
; the epilogue. Its unwind codes then have no pac_sign_lr.

declare swifttailcc void @callee12(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare swifttailcc void @callee1(i64)
declare void @use(ptr)
declare void @clobber()

; In each function below, the prologue signs the return address with pacibsp.
; The return address is then authenticated against the SP it was signed with
; (computed into x16) before the epilogue starts, and the epilogue itself
; neither authenticates nor restores LR. The authentication is a single
; autib with PAuth, or a sequence through x17 in hint space without.

; CHECK-LABEL: pac_grow:
; HINT:          hint #27
; PAUTH:         pacibsp
; CHECK-NEXT:    .seh_pac_sign_lr
; CHECK:         .seh_endprologue
; HINT:          ldr x17, [sp]
; PAUTH:         ldr x30, [sp]
; CHECK:         add x16, sp, #48
; HINT:          hint #14
; HINT-NEXT:     mov x30, x17
; PAUTH:         autib x30, x16
; CHECK:         .seh_startepilogue
; CHECK-NOT:     x30
; CHECK-NOT:     .seh_pac_sign_lr
; CHECK:         add sp, sp, #16
; CHECK:         .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @pac_grow(i64 %a) "sign-return-address"="all" {
  call void @clobber()
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; This function receives stack arguments the tail callee doesn't take.
; CHECK-LABEL: pac_shrink:
; HINT:          hint #27
; PAUTH:         pacibsp
; CHECK-NEXT:    .seh_pac_sign_lr
; CHECK:         .seh_endprologue
; HINT:          ldr x17, [sp]
; PAUTH:         ldr x30, [sp]
; CHECK:         add x16, sp, #16
; HINT:          hint #14
; HINT-NEXT:     mov x30, x17
; PAUTH:         autib x30, x16
; CHECK:         .seh_startepilogue
; CHECK-NOT:     x30
; CHECK-NOT:     .seh_pac_sign_lr
; CHECK:         add sp, sp, #32
; CHECK:         .seh_endepilogue
; CHECK-NEXT:    b callee1
define swifttailcc void @pac_shrink(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64) "sign-return-address"="all" {
  call void @clobber()
  musttail call swifttailcc void @callee1(i64 %a)
  ret void
}

; Callee-popped stack arguments, with a normal return.
; CHECK-LABEL: pac_ret:
; HINT:          hint #27
; PAUTH:         pacibsp
; CHECK-NEXT:    .seh_pac_sign_lr
; CHECK:         .seh_endprologue
; HINT:          ldr x17, [sp]
; PAUTH:         ldr x30, [sp]
; CHECK:         add x16, sp, #16
; HINT:          hint #14
; HINT-NEXT:     mov x30, x17
; PAUTH:         autib x30, x16
; CHECK:         .seh_startepilogue
; CHECK-NOT:     x30
; CHECK-NOT:     .seh_pac_sign_lr
; CHECK:         add sp, sp, #32
; CHECK:         .seh_endepilogue
; CHECK-NEXT:    ret
define swifttailcc void @pac_ret(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64) "sign-return-address"="all" {
  call void @clobber()
  ret void
}

; With dynamic allocation, the frame record is found using the frame pointer.
; CHECK-LABEL: pac_alloca:
; HINT:          hint #27
; PAUTH:         pacibsp
; CHECK-NEXT:    .seh_pac_sign_lr
; CHECK:         .seh_endprologue
; HINT:          ldr x17, [x29, #8]
; PAUTH:         ldr x30, [x29, #8]
; CHECK:         add x16, x29, #48
; HINT:          hint #14
; HINT-NEXT:     mov x30, x17
; PAUTH:         autib x30, x16
; CHECK:         .seh_startepilogue
; CHECK-NOT:     x30
; CHECK-NOT:     .seh_pac_sign_lr
; CHECK:         mov sp, x29
; CHECK-NEXT:    .seh_set_fp
; CHECK:         add sp, sp, #16
; CHECK:         .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @pac_alloca(i64 %a, i64 %n) "sign-return-address"="all" {
  %buf = alloca i64, i64 %n
  call void @use(ptr %buf)
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; The same goes for a stack that is dynamically realigned.
; CHECK-LABEL: pac_align:
; HINT:          hint #27
; PAUTH:         pacibsp
; CHECK-NEXT:    .seh_pac_sign_lr
; CHECK:         .seh_endprologue
; HINT:          ldr x17, [x29, #8]
; PAUTH:         ldr x30, [x29, #8]
; CHECK:         add x16, x29, #48
; HINT:          hint #14
; HINT-NEXT:     mov x30, x17
; PAUTH:         autib x30, x16
; CHECK:         .seh_startepilogue
; CHECK-NOT:     x30
; CHECK-NOT:     .seh_pac_sign_lr
; CHECK:         mov sp, x29
; CHECK-NEXT:    .seh_set_fp
; CHECK:         add sp, sp, #16
; CHECK:         .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @pac_align(i64 %a) "sign-return-address"="all" {
  %buf = alloca [4 x i64], align 64
  call void @use(ptr %buf)
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; The return address is paired with another register, and not at offset 0.
; CHECK-LABEL: pac_csr:
; HINT:          hint #27
; PAUTH:         pacibsp
; CHECK-NEXT:    .seh_pac_sign_lr
; CHECK:         stp x25, x30, [sp, #32]
; CHECK-NEXT:    .seh_save_lrpair x25, 32
; CHECK:         .seh_endprologue
; HINT:          ldr x17, [sp, #40]
; PAUTH:         ldr x30, [sp, #40]
; CHECK:         add x16, sp, #96
; HINT:          hint #14
; HINT-NEXT:     mov x30, x17
; PAUTH:         autib x30, x16
; CHECK:         .seh_startepilogue
; CHECK-NOT:     x30
; CHECK-NOT:     .seh_pac_sign_lr
; CHECK:         ldr x25, [sp, #32]
; CHECK:         add sp, sp, #64
; CHECK:         .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @pac_csr(i64 %a) "sign-return-address"="all" {
  call void asm sideeffect "", "~{x19},~{x21},~{x23},~{x24},~{x25},~{d8},~{d9}"()
  call void @clobber()
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; The return address is never spilled, so it is still in LR. The epilogue
; has no instructions, but its unwind codes still pop the tail callee's
; arguments.
; CHECK-LABEL: pac_leaf:
; HINT:          hint #27
; PAUTH:         pacibsp
; CHECK-NEXT:    .seh_pac_sign_lr
; CHECK:         .seh_endprologue
; HINT:          mov x17, x30
; CHECK:         add x16, sp, #32
; HINT:          hint #14
; HINT-NEXT:     mov x30, x17
; PAUTH:         autib x30, x16
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee12
define swifttailcc void @pac_leaf(i64 %a) "sign-return-address"="all" {
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; The prologue's unwind codes still have pac_sign_lr; the epilogue's don't, and
; they don't restore LR either, as the authenticated return address stays in LR.
; UNWIND-LABEL: Function: pac_grow (
; UNWIND:       Prologue [
; UNWIND:         pacibsp
; UNWIND:       ]
; UNWIND-NEXT:  EpilogueScopes [
; UNWIND-NEXT:    EpilogueScope {
; UNWIND:           Opcodes [
; UNWIND-NEXT:        add sp, #16
; UNWIND-NEXT:        end_c
; UNWIND-NEXT:        add sp, #32
; UNWIND-NEXT:        end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: pac_shrink (
; UNWIND:       Epilogue [
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: pac_ret (
; UNWIND:       Epilogue [
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: pac_alloca (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    mov sp, fp
; UNWIND-NEXT:    ldr x29, [sp, #0]
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: pac_align (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    mov sp, fp
; UNWIND-NEXT:    ldr x29, [sp, #0]
; UNWIND-NEXT:    add sp, #16
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
; UNWIND-LABEL: Function: pac_csr (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    ldp d8, d9, [sp, #48]
; UNWIND-NEXT:    ldr x25, [sp, #32]
; UNWIND-NEXT:    ldp x23, x24, [sp, #16]
; UNWIND-NEXT:    ldr x21, [sp, #8]
; UNWIND-NEXT:    ldr x19, [sp, #0]
; UNWIND-NEXT:    add sp, #64
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #32
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]

; The epilogue has no instructions of its own here, as the tail callee's
; arguments take all of the frame, but its unwind codes must still pop the
; callee's arguments after end_c. They replace the prologue's, which would
; authenticate the return address again.
declare swifttailcc void @callee15(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

; CHECK-LABEL: pac_leaf_grow:
; CHECK:         .seh_endprologue
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_stackalloc 64
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    b callee15
define swifttailcc void @pac_leaf_grow(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) "sign-return-address"="all" {
  musttail call swifttailcc void @callee15(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}

; UNWIND-LABEL: Function: pac_leaf_grow (
; UNWIND:       Opcodes [
; UNWIND-NEXT:    end_c
; UNWIND-NEXT:    add sp, #64
; UNWIND-NEXT:    end
; UNWIND-NEXT:  ]
