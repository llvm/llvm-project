# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK-NEXT: .byte 0
TEST0:  
        .byte 0

# CHECK: TEST1:
# CHECK-NEXT: .short 3
TEST1:  
        .short 3

# CHECK: TEST2:
# CHECK-NEXT: .long 8
TEST2:  
        .long 8

# CHECK: TEST3:
# CHECK-NEXT: .quad 9
TEST3:  
        .quad 9


# rdar://7997827
TEST4:
        .quad 0b0100
        .quad 4294967295
        .quad 4294967295+1
        .quad 4294967295LL+1
        .quad 0b10LL + 07ULL + 0x42AULL
# CHECK: TEST4
# CHECK-NEXT: 	.quad	4
# CHECK-NEXT: 	.quad	4294967295
# CHECK-NEXT: 	.quad	4294967296
# CHECK-NEXT: 	.quad	4294967296
# CHECK-NEXT: 	.quad	1075


TEST5:
        .value 8
# CHECK: TEST5:
# CHECK-NEXT: .short 8

TEST6:
        .byte 'c'
        .byte '\''
        .byte '\\'
        .byte '\#'
        .byte '\t'
        .byte '\n'
        .byte '\r'
        .byte '\f'
        .byte '\"'

# CHECK: TEST6
# CHECK-NEXT:   .byte   99
# CHECK-NEXT:   .byte   39
# CHECK-NEXT:   .byte   92
# CHECK-NEXT:   .byte   35
# CHECK-NEXT:   .byte   9
# CHECK-NEXT:   .byte   10
# CHECK-NEXT:   .byte   13
# CHECK-NEXT:   .byte   12
# CHECK-NEXT:   .byte   34

TEST7:
        .byte 1, 2, 3, 4
# CHECK: TEST7
# CHECK-NEXT:   .byte   1
# CHECK-NEXT:   .byte   2
# CHECK-NEXT:   .byte   3
# CHECK-NEXT:   .byte   4

TEST8:
        .long 0x200000UL+1
        .long 0x200000L+1
# CHECK: TEST8
# CHECK-NEXT: .long 2097153
# CHECK-NEXT: .long 2097153

TEST9:
	.octa 0x1234567812345678abcdef, 340282366920938463463374607431768211455
	.octa 0b00111010010110100101101001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100101101001011010
# CHECK: TEST9
# CHECK-NEXT: .quad 8652035380128501231
# CHECK-NEXT: .quad 1193046
# CHECK-NEXT: .quad -1
# CHECK-NEXT: .quad -1
# CHECK-NEXT: .quad 6510615555426900570
# CHECK-NEXT: .quad 4204772546213206618

