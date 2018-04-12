# Instructions that are valid
#
# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding -show-inst -mcpu=mips3 | FileCheck %s
a:
        .set noat
        abs.d     $f7,$f25             # CHECK: encoding:
        abs.s     $f9,$f16
        add       $s7,$s2,$a1
        add       $9,$14,15176         # CHECK: addi $9, $14, 15176   # encoding: [0x21,0xc9,0x3b,0x48]
        add       $24,-7193            # CHECK: addi $24, $24, -7193  # encoding: [0x23,0x18,0xe3,0xe7]
        add.d     $f1,$f7,$f29
        add.s     $f8,$f21,$f24
        addi      $13,$9,26322
        addi      $8,$8,~1             # CHECK: addi $8, $8, -2 # encoding: [0x21,0x08,0xff,0xfe]
        addu      $9,$a0,$a2
        addu      $9,10                # CHECK: addiu $9, $9, 10    # encoding: [0x25,0x29,0x00,0x0a]
        and       $s7,$v0,$12
        and       $2,4                 # CHECK: andi $2, $2, 4 # encoding: [0x30,0x42,0x00,0x04]
        bc1f      $fcc0, 4             # CHECK: bc1f 4        # encoding: [0x45,0x00,0x00,0x01]
        bc1f      4                    # CHECK: bc1f 4        # encoding: [0x45,0x00,0x00,0x01]
        bc1fl     $fcc0,50             # CHECK: bc1fl 50      # encoding: [0x45,0x02,0x00,0x0c]
        bc1fl     50                   # CHECK: bc1fl 50      # encoding: [0x45,0x02,0x00,0x0c]
        bc1t      $fcc0, 4             # CHECK: bc1t 4        # encoding: [0x45,0x01,0x00,0x01]
        bc1t      4                    # CHECK: bc1t 4        # encoding: [0x45,0x01,0x00,0x01]
        bc1tl     $fcc0,-8239          # CHECK: bc1tl -8239   # encoding: [0x45,0x03,0xf7,0xf4]
        bc1tl     -8239                # CHECK: bc1tl -8239   # encoding: [0x45,0x03,0xf7,0xf4]
        bal       21100                # CHECK: bal 21100     # encoding: [0x04,0x11,0x14,0x9b]
        bgezal    $0, 21100            # CHECK: bal 21100     # encoding: [0x04,0x11,0x14,0x9b]
        bgezal    $6, 21100            # CHECK: bgezal $6, 21100 # encoding: [0x04,0xd1,0x14,0x9b]
        bltzal    $6, 21100            # CHECK: bltzal $6, 21100 # encoding: [0x04,0xd0,0x14,0x9b]
        beql      $14,$s3,12544        # CHECK: beql $14, $19, 12544 # encoding: [0x51,0xd3,0x0c,0x40]
        bgezall   $12,7293             # CHECK: bgezall $12, 7293    # encoding: [0x05,0x93,0x07,0x1f]
        bgezl     $4,-6858             # CHECK: bgezl $4, -6858      # encoding: [0x04,0x83,0xf9,0x4d]
        bgtzl     $10,-3738            # CHECK: bgtzl $10, -3738     # encoding: [0x5d,0x40,0xfc,0x59]
        blezl     $6,2974              # CHECK: blezl $6, 2974       # encoding: [0x58,0xc0,0x02,0xe7]
        bltzall   $6,488               # CHECK: bltzall $6, 488      # encoding: [0x04,0xd2,0x00,0x7a]
        bltzl     $s1,-9964            # CHECK: bltzl $17, -9964     # encoding: [0x06,0x22,0xf6,0x45]
        bnel      $gp,$s4,5107         # CHECK: bnel $gp, $20, 5107  # encoding: [0x57,0x94,0x04,0xfc]
        cache     1, 8($5)             # CHECK: cache 1, 8($5)   # encoding: [0xbc,0xa1,0x00,0x08]
        c.ngl.d   $f29,$f29
        c.ngle.d  $f0,$f16
        c.sf.d    $f30,$f0
        c.sf.s    $f14,$f22
        ceil.l.d  $f1,$f3
        ceil.l.s  $f18,$f13
        ceil.w.d  $f11,$f25
        ceil.w.s  $f6,$f20
        cfc1      $s1,$21
        ctc1      $a2,$26
        cvt.d.l   $f4,$f16
        cvt.d.s   $f22,$f28
        cvt.d.w   $f26,$f11
        cvt.l.d   $f24,$f15
        cvt.l.s   $f11,$f29
        cvt.s.d   $f26,$f8
        cvt.s.l   $f15,$f30
        cvt.s.w   $f22,$f15
        cvt.w.d   $f20,$f14
        cvt.w.s   $f20,$f24
        dadd      $s3,$at,$ra
        dadd      $sp,$s4,-27705       # CHECK: daddi $sp, $20, -27705 # encoding: [0x62,0x9d,0x93,0xc7]
        dadd      $sp,-27705           # CHECK: daddi $sp, $sp, -27705 # encoding: [0x63,0xbd,0x93,0xc7]
        daddi     $sp,$s4,-27705
        daddi     $sp,$s4,-27705       # CHECK: daddi $sp, $20, -27705 # encoding: [0x62,0x9d,0x93,0xc7]
        daddi     $sp,-27705           # CHECK: daddi $sp, $sp, -27705 # encoding: [0x63,0xbd,0x93,0xc7]
        daddiu    $k0,$s6,-4586
        daddu     $s3,$at,$ra
        daddu     $24,$2,18079         # CHECK: daddiu $24, $2, 18079   # encoding: [0x64,0x58,0x46,0x9f]
        daddu     $19,26943            # CHECK: daddiu $19, $19, 26943  # encoding: [0x66,0x73,0x69,0x3f]
        ddiv      $zero,$k0,$s3
        ddivu     $zero,$s0,$s1
        div       $zero,$25,$11
        div.d     $f29,$f20,$f27
        div.s     $f4,$f5,$f15
        divu      $zero,$25,$15
        dmfc1     $12,$f13
        dmtc1     $s0,$f14
        dmult     $s7,$9
        dmultu    $a1,$a2
        dneg      $2                   # CHECK: dneg $2, $2                 # encoding: [0x00,0x02,0x10,0x2e]
        dneg      $2,$3                # CHECK: dneg $2, $3                 # encoding: [0x00,0x03,0x10,0x2e]
        dnegu     $2,$3                # CHECK: dnegu $2, $3                # encoding: [0x00,0x03,0x10,0x2f]
        dsll      $zero,18             # CHECK: dsll $zero, $zero, 18       # encoding: [0x00,0x00,0x04,0xb8]
        dsll      $zero,$s4,18         # CHECK: dsll $zero, $20, 18         # encoding: [0x00,0x14,0x04,0xb8]
        dsll      $zero,$s4,$12        # CHECK: dsllv $zero, $20, $12       # encoding: [0x01,0x94,0x00,0x14]
        dsll      $4, $5               # CHECK: dsllv $4, $4, $5            # encoding: [0x00,0xa4,0x20,0x14]
        dsll      $4, $5, $5           # CHECK: dsllv $4, $5, $5            # encoding: [0x00,0xa5,0x20,0x14]
        dsll32    $zero,18             # CHECK: dsll32 $zero, $zero, 18     # encoding: [0x00,0x00,0x04,0xbc]
        dsll32    $zero,$zero,18       # CHECK: dsll32 $zero, $zero, 18     # encoding: [0x00,0x00,0x04,0xbc]
        dsllv     $zero,$s4,$12        # CHECK: dsllv $zero, $20, $12       # encoding: [0x01,0x94,0x00,0x14]
        dsra      $gp,10               # CHECK: dsra $gp, $gp, 10           # encoding: [0x00,0x1c,0xe2,0xbb]
        dsra      $gp,$s2,10           # CHECK: dsra $gp, $18, 10           # encoding: [0x00,0x12,0xe2,0xbb]
        dsra      $gp,$s2,$s3          # CHECK: dsrav $gp, $18, $19         # encoding: [0x02,0x72,0xe0,0x17]
        dsra32    $gp,10               # CHECK: dsra32 $gp, $gp, 10         # encoding: [0x00,0x1c,0xe2,0xbf]
        dsra32    $gp,$s2,10           # CHECK: dsra32 $gp, $18, 10         # encoding: [0x00,0x12,0xe2,0xbf]
        dsrav     $gp,$s2,$s3          # CHECK: dsrav $gp, $18, $19         # encoding: [0x02,0x72,0xe0,0x17]
        dsrl      $s3,23               # CHECK: dsrl $19, $19, 23           # encoding: [0x00,0x13,0x9d,0xfa]
        dsrl      $s3,$6,23            # CHECK: dsrl $19, $6, 23            # encoding: [0x00,0x06,0x9d,0xfa]
        dsrl      $s3,$6,$s4           # CHECK: dsrlv $19, $6, $20          # encoding: [0x02,0x86,0x98,0x16]
        dsrl      $4, $5               # CHECK: dsrlv $4, $4, $5            # encoding: [0x00,0xa4,0x20,0x16]
        dsrl      $4, $4, $5           # CHECK: dsrlv $4, $4, $5            # encoding: [0x00,0xa4,0x20,0x16]
        dsrl32    $s3,23               # CHECK: dsrl32 $19, $19, 23         # encoding: [0x00,0x13,0x9d,0xfe]
        dsrl32    $s3,$6,23            # CHECK: dsrl32 $19, $6, 23          # encoding: [0x00,0x06,0x9d,0xfe]
        dsrlv     $s3,$6,$s4           # CHECK: dsrlv $19, $6, $20          # encoding: [0x02,0x86,0x98,0x16]
        dsub      $a3,$s6,$8
        dsub      $sp,$s4,-27705       # CHECK: daddi $sp, $20, 27705  # encoding: [0x62,0x9d,0x6c,0x39]
        dsub      $sp,-27705           # CHECK: daddi $sp, $sp, 27705  # encoding: [0x63,0xbd,0x6c,0x39]
        dsubi     $sp,$s4,-27705       # CHECK: daddi $sp, $20, 27705  # encoding: [0x62,0x9d,0x6c,0x39]
        dsubi     $sp,-27705           # CHECK: daddi $sp, $sp, 27705  # encoding: [0x63,0xbd,0x6c,0x39]
        dsubu     $a1,$a1,$k0
        dsubu     $15,$11,5025         # CHECK: daddiu $15, $11, -5025  # encoding: [0x65,0x6f,0xec,0x5f]
        dsubu     $14,-4586            # CHECK: daddiu $14, $14, 4586   # encoding: [0x65,0xce,0x11,0xea]
        ehb                            # CHECK: ehb # encoding:  [0x00,0x00,0x00,0xc0]
        eret
        floor.l.d $f26,$f7
        floor.l.s $f12,$f5
        floor.w.d $f14,$f11
        floor.w.s $f8,$f9
        j         1f                   # CHECK: j .Ltmp0 # encoding: [0b000010AA,A,A,A]
                                       # CHECK:          #   fixup A - offset: 0, value: .Ltmp0, kind: fixup_Mips_26
        j         a                    # CHECK: j a     # encoding: [0b000010AA,A,A,A]
                                       # CHECK:         #   fixup A - offset: 0, value: a, kind: fixup_Mips_26
        j         1328                 # CHECK: j 1328  # encoding: [0x08,0x00,0x01,0x4c]
        jal       21100                # CHECK: jal 21100     # encoding: [0x0c,0x00,0x14,0x9b]
        l.s       $f2, 8($3)           # CHECK: lwc1 $f2, 8($3) # encoding: [0xc4,0x62,0x00,0x08]
        l.d       $f2, 8($3)           # CHECK: ldc1 $f2, 8($3) # encoding: [0xd4,0x62,0x00,0x08]
        lb        $24,-14515($10)
        lbu       $8,30195($v1)
        ld        $sp,-28645($s1)
        ldc1      $f11,16391($s0)
        ldc2      $8,-21181($at)        # CHECK: ldc2 $8, -21181($1)   # encoding: [0xd8,0x28,0xad,0x43]
        ldl       $24,-4167($24)
        ldr       $14,-30358($s4)
        lh        $11,-8556($s5)
        lhu       $s3,-22851($v0)
        li        $at,-29773
        li        $zero,-29889
        ll        $v0,-7321($s2)       # CHECK: ll $2, -7321($18)     # encoding: [0xc2,0x42,0xe3,0x67]
        lld       $zero,-14736($ra)    # CHECK: lld $zero, -14736($ra) # encoding: [0xd3,0xe0,0xc6,0x70]
        lw        $8,5674($a1)
        lwc1      $f16,10225($k0)
        lwc2      $18,-841($a2)        # CHECK: lwc2 $18, -841($6)     # encoding: [0xc8,0xd2,0xfc,0xb7]
        lwl       $s4,-4231($15)
        lwr       $zero,-19147($gp)
        lwu       $s3,-24086($v1)
        mfc1      $a3,$f27
        mfhi      $s3
        mfhi      $sp
        mflo      $s1
        mov.d     $f20,$f14
        mov.s     $f2,$f27
        move      $a0,$a3              # CHECK: move $4, $7              # encoding: [0x00,0xe0,0x20,0x25]
        move      $s5,$a0              # CHECK: move $21, $4             # encoding: [0x00,0x80,0xa8,0x25]
        move      $s8,$a0              # CHECK: move $fp, $4             # encoding: [0x00,0x80,0xf0,0x25]
        move      $25,$a2              # CHECK: move $25, $6             # encoding: [0x00,0xc0,0xc8,0x25]
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
        round.l.d $f12,$f1
        round.l.s $f25,$f5
        round.w.d $f6,$f4
        round.w.s $f27,$f28
        s.s       $f2, 8($3)           # CHECK: swc1  $f2, 8($3)       # encoding: [0xe4,0x62,0x00,0x08]
        s.d       $f2, 8($3)           # CHECK: sdc1  $f2, 8($3)       # encoding: [0xf4,0x62,0x00,0x08]
        sb        $s6,-19857($14)
        sc        $15,18904($s3)       # CHECK: sc $15, 18904($19)     # encoding: [0xe2,0x6f,0x49,0xd8]
        scd       $15,-8243($sp)       # CHECK: scd $15, -8243($sp)    # encoding: [0xf3,0xaf,0xdf,0xcd]
        sd        $12,5835($10)
        sdc1      $f31,30574($13)
        sdc2      $20,23157($s2)       # CHECK: sdc2 $20, 23157($18)   # encoding: [0xfa,0x54,0x5a,0x75]
        sdl       $a3,-20961($s8)
        sdr       $11,-20423($12)
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
        sqrt.d    $f17,$f22
        sqrt.s    $f0,$f1
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
        swl       $15,13694($s3)
        swr       $s1,-26590($14)
        sync                           # CHECK: sync                   # encoding: [0x00,0x00,0x00,0x0f]
        sync 0                         # CHECK: sync                   # encoding: [0x00,0x00,0x00,0x0f]
        syscall                        # CHECK: syscall                # encoding: [0x00,0x00,0x00,0x0c]
        syscall   256                  # CHECK: syscall 256            # encoding: [0x00,0x00,0x40,0x0c]
        teq       $0,$3                # CHECK: teq $zero, $3          # encoding: [0x00,0x03,0x00,0x34]
        teq       $5,$7,620            # CHECK: teq $5, $7, 620        # encoding: [0x00,0xa7,0x9b,0x34]
        teqi      $s5,-17504
        tge       $7,$10               # CHECK: tge $7, $10            # encoding: [0x00,0xea,0x00,0x30]
        tge       $5,$19,340           # CHECK: tge $5, $19, 340       # encoding: [0x00,0xb3,0x55,0x30]
        tgei      $s1,5025
        tgeiu     $sp,-28621
        tgeu      $22,$28              # CHECK: tgeu $22, $gp          # encoding: [0x02,0xdc,0x00,0x31]
        tgeu      $20,$14,379          # CHECK: tgeu $20, $14, 379     # encoding: [0x02,0x8e,0x5e,0xf1]
        tlbp                           # CHECK: tlbp                   # encoding: [0x42,0x00,0x00,0x08]
                                       # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBP
                                       # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBP_MM
        tlbr                           # CHECK: tlbr                   # encoding: [0x42,0x00,0x00,0x01]
                                       # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBR
                                       # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBR_MM
        tlbwi                          # CHECK: tlbwi                  # encoding: [0x42,0x00,0x00,0x02]
                                       # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBWI
                                       # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBWI_MM
        tlbwr                          # CHECK: tlbwr                  # encoding: [0x42,0x00,0x00,0x06]
                                       # CHECK-NEXT:                   # <MCInst #{{[0-9]+}} TLBWR
                                       # CHECK-NOT:                    # <MCInst #{{[0-9]+}} TLBWR_MM
        tlt       $15,$13              # CHECK: tlt $15, $13           # encoding: [0x01,0xed,0x00,0x32]
        tlt       $2,$19,133           # CHECK: tlt $2, $19, 133       # encoding: [0x00,0x53,0x21,0x72]
        tlti      $14,-21059
        tltiu     $ra,-5076
        tltu      $11,$16              # CHECK: tltu $11, $16          # encoding: [0x01,0x70,0x00,0x33]
        tltu      $16,$29,1016         # CHECK: tltu $16, $sp, 1016    # encoding: [0x02,0x1d,0xfe,0x33]
        tne       $6,$17               # CHECK: tne $6, $17            # encoding: [0x00,0xd1,0x00,0x36]
        tne       $7,$8,885            # CHECK: tne $7, $8, 885        # encoding: [0x00,0xe8,0xdd,0x76]
        tnei      $12,-29647
        trunc.l.d $f23,$f23            # CHECK: trunc.l.d $f23, $f23   # encoding: [0x46,0x20,0xbd,0xc9]
        trunc.l.s $f28,$f31            # CHECK: trunc.l.s $f28, $f31   # encoding: [0x46,0x00,0xff,0x09]
        trunc.w.d $f22,$f15            # CHECK: trunc.w.d $f22, $f15   # encoding: [0x46,0x20,0x7d,0x8d]
        trunc.w.s $f28,$f30            # CHECK: trunc.w.s $f28, $f30   # encoding: [0x46,0x00,0xf7,0x0d]
        trunc.w.d $f4,$f6,$4           # CHECK: trunc.w.d $f4, $f6     # encoding: [0x46,0x20,0x31,0x0d]
        trunc.w.s $f4,$f6,$4           # CHECK: trunc.w.s $f4, $f6     # encoding: [0x46,0x00,0x31,0x0d]
        xor       $s2,$a0,$s8
        xor       $2, 4                # CHECK: xori $2, $2, 4         # encoding: [0x38,0x42,0x00,0x04]

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
        ldc1      $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xd4,0x40,A,A]
        swc1      $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xe4,0x40,A,A]
        sdc1      $f0, %lo(g_8)($2)       # CHECK:  encoding: [0xf4,0x40,A,A]
        lwu       $3, %lo(g_8)($2)        # CHECK:  encoding: [0x9c,0x43,A,A]
        ld        $3, %lo(g_8)($2)        # CHECK:  encoding: [0xdc,0x43,A,A]
        sd        $3, %lo(g_8)($2)        # CHECK:  encoding: [0xfc,0x43,A,A]
        ldl       $3, %lo(g_8)($2)        # CHECK:  encoding: [0x68,0x43,A,A]
        ldr       $3, %lo(g_8)($2)        # CHECK:  encoding: [0x6c,0x43,A,A]
        sdl       $3, %lo(g_8)($2)        # CHECK:  encoding: [0xb0,0x43,A,A]
        sdr       $3, %lo(g_8)($2)        # CHECK:  encoding: [0xb4,0x43,A,A]
        .type     g_8,@object
        .comm     g_8,16,16
