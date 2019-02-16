# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips3 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        bc1f      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        bc1t      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        cvt.ps.s  $f3,$f18,$f19   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.s.pl  $f30,$f1        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.s.pu  $f14,$f25       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ldxc1     $f8,$s7($t3)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        luxc1     $f19,$s6($s5)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lwxc1     $f12,$s1($s8)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf      $gp,$8,$fcc0    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf      $gp,$8,$fcc7    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.d    $f6,$f11,$fcc0  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.d    $f6,$f11,$fcc5  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.s    $f23,$f5,$fcc0  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.s    $f23,$f5,$fcc6  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn      $v1,$s1,$s0     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn.d    $f27,$f21,$k0   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn.s    $f12,$f0,$s7    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt      $zero,$s4,$fcc0 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt      $zero,$s4,$fcc5 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.d    $f0,$f2,$fcc0   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.s    $f30,$f2,$fcc0  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.s    $f30,$f2,$fcc1  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz      $a1,$s6,$a5     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.d    $f12,$f29,$a5   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.s    $f25,$f7,$v1    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        pll.ps    $f25,$f9,$f30   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        plu.ps    $f1,$f26,$f29   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdxc1     $f11,$a6($t2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        suxc1     $f12,$k1($t1)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        swxc1     $f19,$t0($k0)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.eq.s    $fcc1, $f2, $f8 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.f.s     $fcc4, $f2, $f7 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.le.s    $fcc6, $f2, $f4 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.lt.s    $fcc2, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.nge.s   $fcc3, $f2, $f8 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngl.s   $fcc2, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngle.s  $fcc2, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngt.s   $fcc5, $f8, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ole.s   $fcc3, $f7, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.olt.s   $fcc6, $f2, $f7 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.seq.s   $fcc7, $f1, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.sf.s    $fcc4, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ueq.s   $fcc6, $f3, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ule.s   $fcc7, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ult.s   $fcc7, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.un.s    $fcc1, $f2, $f4 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.eq.d    $fcc1, $f2, $f8 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.f.d     $fcc4, $f2, $f8 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.le.d    $fcc6, $f2, $f4 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.lt.d    $fcc2, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.nge.d   $fcc3, $f2, $f8 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngl.d   $fcc2, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngle.d  $fcc2, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ngt.d   $fcc5, $f8, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ole.d   $fcc3, $f8, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.olt.d   $fcc6, $f2, $f8 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.seq.d   $fcc7, $f1, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.sf.d    $fcc4, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ueq.d   $fcc6, $f3, $f2 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ule.d   $fcc7, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.ult.d   $fcc7, $f2, $f6 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        c.un.d    $fcc1, $f2, $f4 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level

