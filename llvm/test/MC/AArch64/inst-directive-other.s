// RUN: llvm-mc %s -triple=arm64-apple-darwin -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=arm64-apple-darwin -filetype=obj -o - \
// RUN:   | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-CODE
// RUN: llvm-mc %s -triple=aarch64-win32-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=aarch64-win32-gnu -filetype=obj -o - \
// RUN:   | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-CODE
// RUN: llvm-mc %s -triple=aarch64-linux-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=aarch64-linux-gnu -filetype=obj -o - \
// RUN:   | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-DATA
// RUN: llvm-mc %s -triple=aarch64_be-linux-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=aarch64_be-linux-gnu -filetype=obj -o - \
// RUN:   | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-BE

    .text

    .p2align  2
    .globl _func
_func:
    nop
    // A .long is stored differently for big endian aarch64 targets, while
    // instructions always are stored in little endian.
    // ELF distinguishes between data and code when emitted this way, but
    // MachO and COFF don't.
    .long 0xd503201f
    .inst 0xd503201f

// CHECK-ASM:        .p2align  2
// CHECK-ASM:        .globl  _func
// CHECK-ASM: _func:
// CHECK-ASM:        nop
// CHECK-ASM:        .{{long|word}}   3573751839
// CHECK-ASM:        .inst   0xd503201f

// CHECK-OBJ:        0:       d503201f     nop
// CHECK-OBJ-CODE:   4:       d503201f     nop
// CHECK-OBJ-DATA:   4:       1f 20 03 d5     .word 0xd503201f
// CHECK-OBJ-BE:     4:       d5 03 20 1f     .word 0xd503201f
// CHECK-OBJ:        8:       d503201f     nop
