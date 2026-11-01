// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s
// RUN: llvm-mc -triple aarch64_lfi -filetype=obj %s | llvm-objdump -d - | FileCheck %s --check-prefix=OBJ

.arch_extension pauth

mov x30, x0
ret
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

ldr x30, [sp]
ret
// CHECK:      ldr x30, [sp]
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

ldp x29, x30, [sp]
ret
// CHECK:      ldp x29, x30, [sp]
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

ldp x30, x29, [sp]
ret
// CHECK:      ldp x30, x29, [sp]
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

mov x30, x0
next_func:
nop
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK:      nop

autiasp
ret
// CHECK:      autiasp
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

paciasp
nop
// CHECK:      paciasp
// CHECK-NEXT: nop

mov x30, x0
bl some_func
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: bl some_func

mov x30, x0
blr x1
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: add x28, x27, w1, uxtw
// CHECK-NEXT: blr x28

mov x30, x0
b some_func
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: b some_func

// A temporary label does not flush the deferred LR guard, allowing a
// following authentication instruction to run on the signed pointer.
ldp x29, x30, [sp]
.Ltmp0:
autiasp
ret
// CHECK:      ldp x29, x30, [sp]
// CHECK:      autiasp
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

// DWARF CFI directives (which emit internal temporary labels) also must not
// flush the deferred LR guard before authentication.
.cfi_startproc
ldp x29, x30, [sp], #16
.cfi_def_cfa_offset 0
autiasp
ret
.cfi_endproc
// CHECK:      ldp x29, x30, [sp], #16
// CHECK:      autiasp
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret
// OBJ:        ldp x29, x30, [sp], #0x10
// OBJ-NEXT:   autiasp
// OBJ-NEXT:   add x30, x27, w30, uxtw
// OBJ-NEXT:   ret

mov x30, x0
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
