# Check the hardware registers
#
# FIXME: Use the code generator in order to print the .set directives
#        instead of the instruction printer.
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:      FileCheck %s
        .set noat
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $hwr_cpunum
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x00,0x3b]
        rdhwr     $a0,$hwr_cpunum
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $hwr_cpunum
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x00,0x3b]
        rdhwr     $a0,$0

        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $5, $hwr_synci_step
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x05,0x08,0x3b]
        rdhwr     $a1,$hwr_synci_step
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $5, $hwr_synci_step
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x05,0x08,0x3b]
        rdhwr     $a1,$1

        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $6, $hwr_cc
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x06,0x10,0x3b]
        rdhwr     $a2,$hwr_cc
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $6, $hwr_cc
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x06,0x10,0x3b]
        rdhwr     $a2,$2

        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $7, $hwr_ccres
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x07,0x18,0x3b]
        rdhwr     $a3,$hwr_ccres
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $7, $hwr_ccres
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x07,0x18,0x3b]
        rdhwr     $a3,$3

        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $4
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x20,0x3b]
        rdhwr     $a0,$4
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $5
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x28,0x3b]
        rdhwr     $a0,$5
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $6
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x30,0x3b]
        rdhwr     $a0,$6
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $7
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x38,0x3b]
        rdhwr     $a0,$7
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $8
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x40,0x3b]
        rdhwr     $a0,$8
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $9
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x48,0x3b]
        rdhwr     $a0,$9
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $10
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x50,0x3b]
        rdhwr     $a0,$10
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $11
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x58,0x3b]
        rdhwr     $a0,$11
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $12
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x60,0x3b]
        rdhwr     $a0,$12
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $13
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x68,0x3b]
        rdhwr     $a0,$13
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $14
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x70,0x3b]
        rdhwr     $a0,$14
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $15
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x78,0x3b]
        rdhwr     $a0,$15
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $16
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x80,0x3b]
        rdhwr     $a0,$16
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $17
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x88,0x3b]
        rdhwr     $a0,$17
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $18
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x90,0x3b]
        rdhwr     $a0,$18
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $19
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0x98,0x3b]
        rdhwr     $a0,$19
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $20
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xa0,0x3b]
        rdhwr     $a0,$20
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $21
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xa8,0x3b]
        rdhwr     $a0,$21
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $22
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xb0,0x3b]
        rdhwr     $a0,$22
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $23
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xb8,0x3b]
        rdhwr     $a0,$23
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $24
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xc0,0x3b]
        rdhwr     $a0,$24
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $25
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xc8,0x3b]
        rdhwr     $a0,$25
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $26
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xd0,0x3b]
        rdhwr     $a0,$26
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $27
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xd8,0x3b]
        rdhwr     $a0,$27
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $28
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xe0,0x3b]
        rdhwr     $a0,$28

        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $29
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xe8,0x3b]
        rdhwr     $a0,$29
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $29
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xe8,0x3b]
        rdhwr     $a0,$hwr_ulr

        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $30
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xf0,0x3b]
        rdhwr     $a0,$30
        # CHECK:      .set    push
        # CHECK-NEXT: .set    mips32r2
        # CHECK-NEXT: rdhwr   $4, $31
        # CHECK-NEXT: .set    pop             # encoding: [0x7c,0x04,0xf8,0x3b]
        rdhwr     $a0,$31
