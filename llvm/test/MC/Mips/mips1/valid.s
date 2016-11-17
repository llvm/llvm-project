# Instructions that are valid
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips1 | FileCheck %s
a:
        .set noat
        abs.d     $f7,$f25             # CHECK: encoding:
        abs.s     $f9,$f16
        add       $s7,$s2,$a1
        add.d     $f1,$f7,$f29
        add.s     $f8,$f21,$f24
        addi      $13,$9,26322
        addi      $8,$8,~1             # CHECK: addi $8, $8, -2 # encoding: [0x21,0x08,0xff,0xfe]
        add       $9,$14,15176         # CHECK: addi $9, $14, 15176   # encoding: [0x21,0xc9,0x3b,0x48]
        add       $24,-7193            # CHECK: addi $24, $24, -7193  # encoding: [0x23,0x18,0xe3,0xe7]
        addu      $9,$a0,$a2
        addu      $9,10                # CHECK: addiu $9, $9, 10    # encoding: [0x25,0x29,0x00,0x0a]
        and       $s7,$v0,$12
        and       $2,4                 # CHECK: andi $2, $2, 4 # encoding: [0x30,0x42,0x00,0x04]
        bc1f      $fcc0, 4             # CHECK: bc1f 4        # encoding: [0x45,0x00,0x00,0x01]
        bc1f      4                    # CHECK: bc1f 4        # encoding: [0x45,0x00,0x00,0x01]
        bc1t      $fcc0, 4             # CHECK: bc1t 4        # encoding: [0x45,0x01,0x00,0x01]
        bc1t      4                    # CHECK: bc1t 4        # encoding: [0x45,0x01,0x00,0x01]
        bal       21100                # CHECK: bal 21100     # encoding: [0x04,0x11,0x14,0x9b]
        bgezal    $0, 21100            # CHECK: bal 21100     # encoding: [0x04,0x11,0x14,0x9b]
        bgezal    $6, 21100            # CHECK: bgezal $6, 21100 # encoding: [0x04,0xd1,0x14,0x9b]
        bltzal    $6, 21100            # CHECK: bltzal $6, 21100 # encoding: [0x04,0xd0,0x14,0x9b]
        c.ngl.d   $f29,$f29
        c.ngle.d  $f0,$f16
        c.sf.d    $f30,$f0
        c.sf.s    $f14,$f22
        cfc1      $s1,$21
        ctc1      $a2,$26
        cvt.d.s   $f22,$f28
        cvt.d.w   $f26,$f11
        cvt.s.d   $f26,$f8
        cvt.s.w   $f22,$f15
        cvt.w.d   $f20,$f14
        cvt.w.s   $f20,$f24
        div       $zero,$25,$11
        div.d     $f29,$f20,$f27
        div.s     $f4,$f5,$f15
        divu      $zero,$25,$15
        ehb                            # CHECK: ehb # encoding:  [0x00,0x00,0x00,0xc0]
        j         1f                   # CHECK: j $tmp0 # encoding: [0b000010AA,A,A,A]
                                       # CHECK:         #   fixup A - offset: 0, value: ($tmp0), kind: fixup_Mips_26
        j         a                    # CHECK: j a     # encoding: [0b000010AA,A,A,A]
                                       # CHECK:         #   fixup A - offset: 0, value: a, kind: fixup_Mips_26
        j         1328                 # CHECK: j 1328  # encoding: [0x08,0x00,0x01,0x4c]
        jal       21100                # CHECK: jal 21100     # encoding: [0x0c,0x00,0x14,0x9b]
        lb        $24,-14515($10)
        lbu       $8,30195($v1)
        lh        $11,-8556($s5)
        lhu       $s3,-22851($v0)
        li        $at,-29773
        li        $zero,-29889
        lw        $8,5674($a1)
        lwc1      $f16,10225($k0)
        lwc2      $18,-841($a2)        # CHECK: lwc2 $18, -841($6)     # encoding: [0xc8,0xd2,0xfc,0xb7]
        lwc3      $10,-32265($k0)
        lwl       $s4,-4231($15)
        lwr       $zero,-19147($gp)
        mfc1      $a3,$f27
        mfhi      $s3
        mfhi      $sp
        mflo      $s1
        mov.d     $f20,$f14
        mov.s     $f2,$f27
        move      $s8,$a0              # CHECK: move $fp, $4           # encoding: [0x00,0x80,0xf0,0x25]
        move      $25,$a2              # CHECK: move $25, $6           # encoding: [0x00,0xc0,0xc8,0x25]
        mtc1      $s8,$f9
        mthi      $s1
        mtlo      $sp
        mtlo      $25
        mul.d     $f20,$f20,$f16
        mul.s     $f30,$f10,$f2
        mult      $sp,$s4
        mult      $sp,$v0
        multu     $gp,$k0
        multu     $9,$s2
        neg       $2                   # CHECK: neg  $2, $2            # encoding: [0x00,0x02,0x10,0x22]
        neg       $2, $3               # CHECK: neg  $2, $3            # encoding: [0x00,0x03,0x10,0x22]
        negu      $2                   # CHECK: negu $2, $2            # encoding: [0x00,0x02,0x10,0x23]
        negu      $2,$3                # CHECK: negu $2, $3            # encoding: [0x00,0x03,0x10,0x23]
        neg.d     $f27,$f18
        neg.s     $f1,$f15
        nop
        nor       $a3,$zero,$a3
        not       $3, $4               # CHECK: not $3, $4             # encoding: [0x00,0x80,0x18,0x27]
        not       $3                   # CHECK: not $3, $3             # encoding: [0x00,0x60,0x18,0x27]
        or        $12,$s0,$sp
        or        $2, 4                # CHECK: ori $2, $2, 4          # encoding: [0x34,0x42,0x00,0x04]
        sb        $s6,-19857($14)
        sgt       $4, $5               # CHECK: slt $4, $5, $4         # encoding: [0x00,0xa4,0x20,0x2a]
        sgt       $4, $5, $6           # CHECK: slt $4, $6, $5         # encoding: [0x00,0xc5,0x20,0x2a]
        sgtu      $4, $5               # CHECK: sltu $4, $5, $4        # encoding: [0x00,0xa4,0x20,0x2b]
        sgtu      $4, $5, $6           # CHECK: sltu $4, $6, $5        # encoding: [0x00,0xc5,0x20,0x2b]
        sh        $14,-6704($15)
        sll       $4, $5               # CHECK: sllv $4, $4, $5        # encoding: [0x00,0xa4,0x20,0x04]
        sll       $a3,18               # CHECK: sll $7, $7, 18         # encoding: [0x00,0x07,0x3c,0x80]
        sll       $a3,$zero,18         # CHECK: sll $7, $zero, 18      # encoding: [0x00,0x00,0x3c,0x80]
        sll       $a3,$zero,$9         # CHECK: sllv $7, $zero, $9     # encoding: [0x01,0x20,0x38,0x04]
        sllv      $a3,$zero,$9         # CHECK: sllv $7, $zero, $9     # encoding: [0x01,0x20,0x38,0x04]
        slt       $s7,$11,$k1          # CHECK: slt $23, $11, $27      # encoding: [0x01,0x7b,0xb8,0x2a]
        slti      $s1,$10,9489         # CHECK: slti $17, $10, 9489    # encoding: [0x29,0x51,0x25,0x11]
        sltiu     $25,$25,-15531       # CHECK: sltiu $25, $25, -15531 # encoding: [0x2f,0x39,0xc3,0x55]
        sltu      $s4,$s5,$11          # CHECK: sltu  $20, $21, $11    # encoding: [0x02,0xab,0xa0,0x2b]
        sltu      $24,$25,-15531       # CHECK: sltiu $24, $25, -15531 # encoding: [0x2f,0x38,0xc3,0x55]
        sra       $4, $5               # CHECK: srav $4, $4, $5        # encoding: [0x00,0xa4,0x20,0x07]
        sra       $s1,15               # CHECK: sra $17, $17, 15       # encoding: [0x00,0x11,0x8b,0xc3]
        sra       $s1,$s7,15           # CHECK: sra $17, $23, 15       # encoding: [0x00,0x17,0x8b,0xc3]
        sra       $s1,$s7,$sp          # CHECK: srav $17, $23, $sp     # encoding: [0x03,0xb7,0x88,0x07]
        srav      $s1,$s7,$sp          # CHECK: srav $17, $23, $sp     # encoding: [0x03,0xb7,0x88,0x07]
        srl       $4, $5               # CHECK: srlv $4, $4, $5        # encoding: [0x00,0xa4,0x20,0x06]
        srl       $2,7                 # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
        srl       $2,$2,7              # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
        srl       $25,$s4,$a0          # CHECK: srlv $25, $20, $4      # encoding: [0x00,0x94,0xc8,0x06]
        srlv      $25,$s4,$a0          # CHECK: srlv $25, $20, $4      # encoding: [0x00,0x94,0xc8,0x06]
        ssnop                          # CHECK: ssnop                  # encoding: [0x00,0x00,0x00,0x40]
        sub       $s6,$s3,$12
        sub       $22,$17,-3126        # CHECK: addi $22, $17, 3126    # encoding: [0x22,0x36,0x0c,0x36]
        sub       $13,6512             # CHECK: addi $13, $13, -6512   # encoding: [0x21,0xad,0xe6,0x90]
        sub.d     $f18,$f3,$f17
        sub.s     $f23,$f22,$f22
        subu      $sp,$s6,$s6
        sw        $ra,-10160($sp)
        swc1      $f6,-8465($24)
        swc2      $25,24880($s0)       # CHECK: swc2 $25, 24880($16)   # encoding: [0xea,0x19,0x61,0x30]
        swc3      $10,-32265($k0)
        swl       $15,13694($s3)
        swr       $s1,-26590($14)
        syscall                        # CHECK: syscall                # encoding: [0x00,0x00,0x00,0x0c]
        syscall   256                  # CHECK: syscall 256            # encoding: [0x00,0x00,0x40,0x0c]
        tlbp                           # CHECK: tlbp                   # encoding: [0x42,0x00,0x00,0x08]
        tlbr                           # CHECK: tlbr                   # encoding: [0x42,0x00,0x00,0x01]
        tlbwi                          # CHECK: tlbwi                  # encoding: [0x42,0x00,0x00,0x02]
        tlbwr                          # CHECK: tlbwr                  # encoding: [0x42,0x00,0x00,0x06]
        xor       $s2,$a0,$s8
        xor       $2, 4                # CHECK: xori $2, $2, 4         # encoding: [0x38,0x42,0x00,0x04]

        .set at
        trunc.w.s  $f4,$f6,$4
        # CHECK:                cfc1    $4, $ra                 # encoding: [0x44,0x44,0xf8,0x00]
        # CHECK:                cfc1    $4, $ra                 # encoding: [0x44,0x44,0xf8,0x00]
        # CHECK:                nop                             # encoding: [0x00,0x00,0x00,0x00]
        # CHECK:                ori     $1, $4, 3               # encoding: [0x34,0x81,0x00,0x03]
        # CHECK:                xori    $1, $1, 2               # encoding: [0x38,0x21,0x00,0x02]
        # CHECK:                ctc1    $1, $ra                 # encoding: [0x44,0xc1,0xf8,0x00]
        # CHECK:                nop                             # encoding: [0x00,0x00,0x00,0x00]
        # CHECK:                cvt.w.s $f4, $f6                # encoding: [0x46,0x00,0x31,0x24]
        # CHECK:                ctc1    $4, $ra                 # encoding: [0x44,0xc4,0xf8,0x00]
        # CHECK:                nop                             # encoding: [0x00,0x00,0x00,0x00]

        trunc.w.d  $f4,$f6,$4
        # CHECK:                cfc1    $4, $ra                 # encoding: [0x44,0x44,0xf8,0x00]
        # CHECK:                cfc1    $4, $ra                 # encoding: [0x44,0x44,0xf8,0x00]
        # CHECK:                nop                             # encoding: [0x00,0x00,0x00,0x00]
        # CHECK:                ori     $1, $4, 3               # encoding: [0x34,0x81,0x00,0x03]
        # CHECK:                xori    $1, $1, 2               # encoding: [0x38,0x21,0x00,0x02]
        # CHECK:                ctc1    $1, $ra                 # encoding: [0x44,0xc1,0xf8,0x00]
        # CHECK:                nop                             # encoding: [0x00,0x00,0x00,0x00]
        # CHECK:                cvt.w.d $f4, $f6                # encoding: [0x46,0x20,0x31,0x24]
        # CHECK:                ctc1    $4, $ra                 # encoding: [0x44,0xc4,0xf8,0x00]
        # CHECK:                nop                             # encoding: [0x00,0x00,0x00,0x00]

1:
        # Check that we accept traditional %relocation(symbol) offsets for stores
        # and loads, not just a sign 16 bit offset.

        lui       $2, %hi(g_8)            # CHECK:  encoding: [0x3c,0x02,A,A]
        lb        $3, %lo(g_8)($2)        # CHECK:  encoding: [0x80,0x43,A,A]
        lh        $3, %lo(g_8)($2)        # CHECK:  encoding: [0x84,0x43,A,A]
        lhu       $3, %lo(g_8)($2)        # CHECK:  encoding: [0x94,0x43,A,A]
        lw        $3, %lo(g_8)($2)        # CHECK:  encoding: [0x8c,0x43,A,A]
        sb        $3, %lo(g_8)($2)        # CHECK:  encoding: [0xa0,0x43,A,A]
        sh        $3, %lo(g_8)($2)        # CHECK:  encoding: [0xa4,0x43,A,A]
        sw        $3, %lo(g_8)($2)        # CHECK:  encoding: [0xac,0x43,A,A]

        lwl       $3, %lo(g_8)($2)        # CHECK:  encoding: [0x88,0x43,A,A]
        lwr       $3, %lo(g_8)($2)        # CHECK:  encoding: [0x98,0x43,A,A]
        swl       $3, %lo(g_8)($2)        # CHECK:  encoding: [0xa8,0x43,A,A]
        swr       $3, %lo(g_8)($2)        # CHECK:  encoding: [0xb8,0x43,A,A]

        lwc1      $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xc4,0x40,A,A]
        swc1      $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xe4,0x40,A,A]
        .type     g_8,@object
        .comm     g_8,16,16
