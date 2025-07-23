// RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s -o - | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readelf -SW - | FileCheck --check-prefix=OBJ %s

// Minimum alignment = preferred alignment, no SHT_LLVM_MIN_ADDRALIGN needed.
// ASM: .section .text.f1
// ASM: .p2align 2
// ASM: .prefalign 4 
// OBJ: .text.f1
// OBJ-NOT: .llvm.minalign
.section .text.f1,"ax",@progbits
.p2align 2
.prefalign 4

// Minimum alignment < preferred alignment, SHT_LLVM_MIN_ADDRALIGN emitted.
// ASM: .section .text.f2
// ASM: .p2align 2
// ASM: .prefalign 8 
// OBJ: [ 4] .text.f2          PROGBITS           0000000000000000 000040 000000 00  AX  0   0  8
// OBJ: [ 5] .llvm.minalign    LLVM_MIN_ADDRALIGN 0000000000000000 000000 000000 00  LE  4   0  4
.section .text.f2,"ax",@progbits
.p2align 2
.prefalign 8

// Minimum alignment > preferred alignment, preferred alignment rounded up to
// minimum alignment. No SHT_LLVM_MIN_ADDRALIGN emitted.
// ASM: .section .text.f3
// ASM: .p2align 3
// ASM: .prefalign 4 
// OBJ: .text.f3
// OBJ-NOT: .llvm.minalign
.section .text.f3,"ax",@progbits
.p2align 3
.prefalign 4

// Maximum of all .prefalign directives written to object file.
// ASM: .section .text.f4
// ASM: .p2align 2
// ASM: .prefalign 8
// ASM: .prefalign 16
// ASM: .prefalign 8
// OBJ: [ 7] .text.f4          PROGBITS           0000000000000000 000040 000000 00  AX  0   0 16
// OBJ: [ 8] .llvm.minalign    LLVM_MIN_ADDRALIGN 0000000000000000 000000 000000 00  LE  7   0  4
.section .text.f4,"ax",@progbits
.p2align 2
.prefalign 8
.prefalign 16
.prefalign 8
