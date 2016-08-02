// RUN: llvm-mc %s -triple=aarch64-none-linux-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=aarch64-none-linux-gnu -filetype=obj -o - \
// RUN:   | llvm-readobj -s -sd | FileCheck %s  --check-prefix=CHECK-OBJ

    .section    .inst.aarch64_inst

    .align  2
    .global aarch64_inst
    .type   aarch64_inst,%function
aarch64_inst:
    .inst 0x5e104020

// CHECK-ASM:        .align  2
// CHECK-ASM:        .globl  aarch64_inst
// CHECK-ASM:        .type   aarch64_inst,@function
// CHECK-ASM: aarch64_inst:
// CHECK-ASM:        .inst   0x5e104020

// CHECK-OBJ: Section {
// CHECK-OBJ:   Name: .inst.aarch64_inst
// CHECK-OBJ:   SectionData (
// CHECK-OBJ-NEXT: 0000: 2040105E
// CHECK-OBJ-NEXT: )
