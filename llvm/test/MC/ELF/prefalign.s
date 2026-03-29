// RUN: llvm-mc -triple x86_64 %s -o - | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -filetype=obj -triple x86_64 %s -o - | llvm-readelf -SW - | FileCheck --check-prefix=OBJ %s

// Minimum alignment >= preferred alignment, no effect on sh_addralign.
// ASM: .section .text.f1lt
// ASM: .p2align 2
// ASM: .prefalign 2 
// OBJ: .text.f1lt        PROGBITS        0000000000000000 000040 000003 00  AX  0   0  4
.section .text.f1lt,"ax",@progbits
.p2align 2
.prefalign 2
.rept 3
nop
.endr

// ASM: .section .text.f1eq
// ASM: .p2align 2
// ASM: .prefalign 2 
// OBJ: .text.f1eq        PROGBITS        0000000000000000 000044 000004 00  AX  0   0  4
.section .text.f1eq,"ax",@progbits
.p2align 2
.prefalign 2
.rept 4
nop
.endr

// ASM: .section .text.f1gt
// ASM: .p2align 2
// ASM: .prefalign 2 
// OBJ: .text.f1gt        PROGBITS        0000000000000000 000048 000005 00  AX  0   0  4
.section .text.f1gt,"ax",@progbits
.p2align 2
.prefalign 2
.rept 5
nop
.endr

// Minimum alignment < preferred alignment, sh_addralign influenced by section size.
// Use maximum of all .prefalign directives.
// ASM: .section .text.f2lt
// ASM: .p2align 2
// ASM: .prefalign 8
// ASM: .prefalign 16 
// ASM: .prefalign 8
// OBJ: .text.f2lt        PROGBITS        0000000000000000 000050 000003 00  AX  0   0  4
.section .text.f2lt,"ax",@progbits
.p2align 2
.prefalign 8
.prefalign 16
.prefalign 8
.rept 3
nop
.endr

// ASM: .section .text.f2between1
// OBJ: .text.f2between1  PROGBITS        0000000000000000 000054 000008 00  AX  0   0  8
.section .text.f2between1,"ax",@progbits
.p2align 2
.prefalign 8
.prefalign 16
.prefalign 8
.rept 8
nop
.endr

// OBJ: .text.f2between2  PROGBITS        0000000000000000 00005c 000009 00  AX  0   0 16
.section .text.f2between2,"ax",@progbits
.p2align 2
.prefalign 8
.prefalign 16
.prefalign 8
.rept 9
nop
.endr

// OBJ: .text.f2between3  PROGBITS        0000000000000000 000068 000010 00  AX  0   0 16
.section .text.f2between3,"ax",@progbits
.p2align 2
.prefalign 8
.prefalign 16
.prefalign 8
.rept 16
nop
.endr

// OBJ: .text.f2gt1       PROGBITS        0000000000000000 000078 000011 00  AX  0   0 16
.section .text.f2gt1,"ax",@progbits
.p2align 2
.prefalign 8
.prefalign 16
.prefalign 8
.rept 17
nop
.endr

// OBJ: .text.f2gt2       PROGBITS        0000000000000000 00008c 000021 00  AX  0   0 16
.section .text.f2gt2,"ax",@progbits
.p2align 2
.prefalign 8
.prefalign 16
.prefalign 8
.rept 33
nop
.endr
