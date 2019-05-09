// RUN: llvm-mc %s -triple=aarch64-none-linux-gnu -filetype=asm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-mc %s -triple=aarch64-none-linux-gnu -filetype=obj -o %t
// RUN: llvm-readobj -S --sd %t | FileCheck %s  --check-prefix=CHECK-OBJ
// RUN: llvm-objdump -t %t | FileCheck %s  --check-prefix=CHECK-SYMS

    .section    .size.aarch64_size

    .p2align  2
    .global aarch64_size
    .type   aarch64_size,%function
aarch64_size:
    .hword half_word
    .word  full_word
    .dword double_word
    .xword also_double_word

// CHECK-ASM:        .p2align  2
// CHECK-ASM:        .globl  aarch64_size
// CHECK-ASM:        .type   aarch64_size,@function
// CHECK-ASM: aarch64_size:
// CHECK-ASM:        .hword half_word
// CHECK-ASM:        .word  full_word
// CHECK-ASM:        .xword double_word
// CHECK-ASM:       .xword also_double_word

// CHECK-OBJ: Section {
// CHECK-OBJ:   Name: .size.aarch64_size
// CHECK-OBJ:   SectionData (
// CHECK-OBJ-NEXT:   0000: 00000000 00000000 00000000 00000000  |................|
// CHECK-OBJ-NEXT:   0010: 00000000 0000                        |......|
// CHECK-OBJ-NEXT: )

// CHECK-SYMS:     0000000000000000         .size.aarch64_size	 00000000 $d.0
// CHECK-SYMS:     0000000000000000 g     F .size.aarch64_size	 00000000 aarch64_size
// CHECK-SYMS:     0000000000000000         *UND*		 00000000 also_double_word
// CHECK-SYMS:     0000000000000000         *UND*		 00000000 double_word
// CHECK-SYMS:     0000000000000000         *UND*		 00000000 full_word
// CHECK-SYMS:     0000000000000000         *UND*		 00000000 half_word
