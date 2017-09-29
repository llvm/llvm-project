# Verify that every branch and jump instruction is followed by a delay slot.
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 | FileCheck %s

        .set noat
        # CHECK: b 1332
        # CHECK: nop
        b 1332
        # CHECK: bc1f 1332
        # CHECK: nop
        bc1f 1332
        # CHECK: bc1t 1332
        # CHECK: nop
        bc1t 1332
        # CHECK: beq $9, $6, 1332
        # CHECK: nop
        beq $9,$6,1332
        # CHECK: bgez $6, 1332
        # CHECK: nop
        bgez $6,1332
        # CHECK: bgezal $6, 1332
        # CHECK: nop
        bgezal $6,1332
        # CHECK: bgtz $6, 1332
        # CHECK: nop
        bgtz $6,1332
        # CHECK: blez $6, 1332
        # CHECK: nop
        blez $6,1332
        # CHECK: bltz $6, 1332
        # CHECK: nop
        bltz $6,1332
        # CHECK: bne $9, $6, 1332
        # CHECK: nop
        bne $9,$6,1332
        # CHECK: bltzal $6, 1332
        # CHECK: nop
        bltzal $6,1332
        # CHECK: bal 1332
        # CHECK: nop
        bal 1332
        # CHECK: bnez $11, 1332
        # CHECK: nop
        bnez $11,1332
        # CHECK: beqz $11, 1332
        # CHECK: nop
        beqz $11,1332

        # CHECK: bc1fl 1332
        # CHECK: nop
        bc1fl 1332
        # CHECK: bc1fl 1332
        # CHECK: nop
        bc1fl $fcc0, 1332
        # CHECK: bc1fl $fcc3, 1332
        # CHECK: nop
        bc1fl $fcc3, 1332
        # CHECK: bc1tl 1332
        # CHECK: nop
        bc1tl 1332
        # CHECK: bc1tl 1332
        # CHECK: nop
        bc1tl $fcc0, 1332
        # CHECK: bc1tl $fcc3, 1332
        # CHECK: nop
        bc1tl $fcc3, 1332
        # CHECK: beql $9, $6, 1332
        # CHECK: nop
        beql $9,$6,1332
        # CHECK: beql $9, $zero, 1332
        # CHECK: nop
        beqzl $9,1332
        # CHECK: bnel $9, $6, 1332
        # CHECK: nop
        bnel $9,$6,1332
        # CHECK: bnel $9, $zero, 1332
        # CHECK: nop
        bnezl $9,1332
        # CHECK: bgezl $6, 1332
        # CHECK: nop
        bgezl $6,1332
        # CHECK: bgtzl $6, 1332
        # CHECK: nop
        bgtzl $6,1332
        # CHECK: blezl $6, 1332
        # CHECK: nop
        blezl $6,1332
        # CHECK: bltzl $6, 1332
        # CHECK: nop
        bltzl $6,1332
        # CHECK: bgezall $6, 1332
        # CHECK: nop
        bgezall $6,1332
        # CHECK: bltzall $6, 1332
        # CHECK: nop
        bltzall $6,1332

        # CHECK: j 1328
        # CHECK: nop
        j 1328
        # CHECK: jal 1328
        # CHECK: nop
        jal 1328
        # CHECK: jalr $6
        # CHECK: nop
        jalr $6
        # CHECK: jalr $25
        # CHECK: nop
        jalr $31,$25
        # CHECK: jalr $10, $11
        # CHECK: nop
        jalr $10,$11
        # CHECK: jr $7
        # CHECK: nop
        jr $7
        # CHECK: jr $7
        # CHECK: nop
        j $7
        # CHECK: jalr $25
        # CHECK: nop
        jal $25
        # CHECK: jalr $4, $25
        # CHECK: nop
        jal $4,$25
        # CHECK: jalx lab
        # CHECK: nop
        jalx lab
