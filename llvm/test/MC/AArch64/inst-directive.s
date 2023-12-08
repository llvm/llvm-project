// RUN: llvm-mc %s -triple=aarch64-none-linux-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=aarch64-none-linux-gnu -filetype=obj -o %t
// RUN: llvm-readobj -S --sd %t | FileCheck %s  --check-prefix=CHECK-OBJ
// RUN: llvm-objdump -t %t | FileCheck %s  --check-prefix=CHECK-SYMS

// RUN: llvm-mc %s -triple=aarch64_be-none-linux-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=aarch64_be-none-linux-gnu -filetype=obj -o %t
// RUN: llvm-readobj -S --sd %t | FileCheck %s  --check-prefix=CHECK-OBJ
// RUN: llvm-objdump -t %t | FileCheck %s  --check-prefix=CHECK-SYMS

    .section    .inst.aarch64_inst

    .p2align  2
    .global aarch64_inst
    .type   aarch64_inst,%function
aarch64_inst:
    .inst 0x5e104020

// CHECK-ASM:        .p2align  2
// CHECK-ASM:        .globl  aarch64_inst
// CHECK-ASM:        .type   aarch64_inst,@function
// CHECK-ASM: aarch64_inst:
// CHECK-ASM:        .inst   0x5e104020

// CHECK-OBJ: Section {
// CHECK-OBJ:   Name: .inst.aarch64_inst
// CHECK-OBJ:   SectionData (
// CHECK-OBJ-NEXT: 0000: 2040105E
// CHECK-OBJ-NEXT: )

// CHECK-SYMS-NOT: 0000000000000000 l .inst.aarch64_inst 0000000000000000 $d
// CHECK-SYMS:     0000000000000000 l .inst.aarch64_inst 0000000000000000 $x
// CHECK-SYMS-NOT: 0000000000000000 l .inst.aarch64_inst 0000000000000000 $d
