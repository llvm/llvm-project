// RUN: llvm-mc %s -triple=armv7-apple-darwin -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=armv7-apple-darwin -filetype=obj -o - \
// RUN:   | llvm-objdump --triple=thumbv7 -d - | FileCheck %s --check-prefixes=CHECK-OBJ-CODE
// RUN: llvm-mc %s -triple=thumbv7-win32-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=thumbv7-win32-gnu -filetype=obj -o - \
// RUN:   | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-CODE
// RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o - \
// RUN:   | llvm-objdump -d --triple=thumbv7 - | FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-DATA

    .text

    .p2align  2
    .globl _func
    .thumb
_func:
    // ELF distinguishes between data and code when emitted this way, but
    // MachO and COFF don't.
    bx      lr
    .short  0x4770
    .inst.n 0x4770
    mov.w   r0, #42
    .short  0xf04f, 0x002a
    .inst.w 0xf04f002a

// CHECK-ASM:        .p2align  2
// CHECK-ASM:        .globl  _func
// CHECK-ASM: _func:
// CHECK-ASM:        bx      lr
// CHECK-ASM:        .short  18288
// CHECK-ASM:        .inst.n 0x4770
// CHECK-ASM:        mov.w   r0, #42
// CHECK-ASM:        .short  61519
// CHECK-ASM:        .short  42
// CHECK-ASM:        .inst.w 0xf04f002a

// CHECK-OBJ:        0:       70 47           bx lr
// CHECK-OBJ-CODE:   2:       70 47           bx lr
// CHECK-OBJ-DATA:   2:       70 47           .short 0x4770
// CHECK-OBJ:        4:       70 47           bx lr
// CHECK-OBJ:        6:       4f f0 2a 00     mov.w   r0, #42
// CHECK-OBJ-CODE:   a:       4f f0 2a 00     mov.w   r0, #42
// CHECK-OBJ-DATA:   a:       4f f0 2a 00     .word 0x002af04f
// CHECK-OBJ:        e:       4f f0 2a 00     mov.w   r0, #42
