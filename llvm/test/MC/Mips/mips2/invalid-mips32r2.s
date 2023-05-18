# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        bc1f      $fcc1, 4          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        bc1t      $fcc1, 4          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        clo     $t3,$a1             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        clz     $sp,$gp             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.l.d $f24,$f15           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.l.s $f11,$f29           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.eq.s    $fcc1, $f2, $f8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.f.s     $fcc4, $f2, $f7   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.le.s    $fcc6, $f2, $f4   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.lt.s    $fcc2, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.nge.s   $fcc3, $f2, $f8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngl.s   $fcc2, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngle.s  $fcc2, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngt.s   $fcc5, $f8, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ole.s   $fcc3, $f7, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.olt.s   $fcc6, $f2, $f7   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.seq.s   $fcc7, $f1, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.sf.s    $fcc4, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ueq.s   $fcc6, $f3, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ule.s   $fcc7, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ult.s   $fcc7, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.un.s    $fcc1, $f2, $f4   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.eq.d    $fcc1, $f2, $f8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.f.d     $fcc4, $f2, $f8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.le.d    $fcc6, $f2, $f4   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.lt.d    $fcc2, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.nge.d   $fcc3, $f2, $f8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngl.d   $fcc2, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngle.d  $fcc2, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngt.d   $fcc5, $f8, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ole.d   $fcc3, $f8, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.olt.d   $fcc6, $f2, $f8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.seq.d   $fcc7, $f1, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.sf.d    $fcc4, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ueq.d   $fcc6, $f3, $f2   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ule.d   $fcc7, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ult.d   $fcc7, $f2, $f6   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.un.d    $fcc1, $f2, $f4   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        deret                       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        di      $s8                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        di                          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ei      $t6                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ei                          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        eret                        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ldxc1   $f8,$s7($t7)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        luxc1   $f19,$s6($s5)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lwxc1   $f12,$s1($s8)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        madd    $s6,$t5             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        madd    $zero,$t1           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        madd.d  $f18,$f19,$f26,$f20 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        madd.s  $f1,$f31,$f19,$f25  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        maddu   $s3,$gp             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        maddu   $t8,$s2             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mfc0    $a2,$14,1           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: selector must be zero for pre-MIPS32 ISAs
        mfhc1   $s8,$f24            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf    $gp,$8,$fcc0        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf    $gp,$8,$fcc7        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.d  $f6,$f11,$fcc0      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.d  $f6,$f11,$fcc5      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.s  $f23,$f5,$fcc0      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.s  $f23,$f5,$fcc6      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn    $v1,$s1,$s0         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn.d  $f27,$f21,$k0       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn.s  $f12,$f0,$s7        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt    $zero,$s4,$fcc0     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt    $zero,$s4,$fcc5     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.d  $f0,$f2,$fcc0       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.s  $f30,$f2,$fcc0      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.s  $f30,$f2,$fcc1      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz    $a1,$s6,$t1         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.d  $f12,$f29,$t1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.s  $f25,$f7,$v1        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msub    $s7,$k1             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msub.d  $f10,$f1,$f31,$f18  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msub.s  $f12,$f19,$f10,$f16 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msubu   $t7,$a1             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mtc0    $t1,$29,3           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: selector must be zero for pre-MIPS32 ISAs
        mthc1   $zero,$f16          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mul     $s0,$s4,$at         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        nmadd.d $f18,$f9,$f14,$f19  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        nmadd.s $f0,$f5,$f25,$f12   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        nmsub.d $f30,$f8,$f16,$f30  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        nmsub.s $f1,$f24,$f19,$f4   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        pause                       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        rotr    $1,15               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        rotr    $1,$14,15           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        rotrv   $1,$14,$15          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdxc1   $f11,$t2($t6)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        seb     $t9,$t7             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        seh     $v1,$t4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        suxc1   $f12,$k1($t5)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        swxc1   $f19,$t4($k0)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        wsbh    $k1,$t1             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ext     $1, $2, 4, 5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ins     $1, $2, 4, 5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
