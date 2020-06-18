# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        bc1f      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        bc1t      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        ceil.l.d  $f1,$f3         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ceil.l.s  $f18,$f13       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.d.l   $f4,$f16        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.l.d   $f24,$f15       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.l.s   $f11,$f29       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.s.l   $f15,$f30       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.ps.s  $f3,$f18,$f19   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.s.pl  $f30,$f1        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.s.pu  $f14,$f25       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dadd      $s3,$at,$ra     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        daddi     $sp,$s4,-27705  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        daddiu    $k0,$s6,-4586   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        daddu     $s3,$at,$ra     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ddiv      $zero,$k0,$s3   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ddivu     $zero,$s0,$s1   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dmfc1     $t0,$f13        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dmtc1     $s0,$f14        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dmultu    $a1,$a2         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsll      $zero,18        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsll      $zero,$s4,18    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsll      $zero,$s4,$t0   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsll32    $zero,18        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsll32    $zero,$zero,18  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsllv     $zero,$s4,$t0   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsra      $gp,10          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsra      $gp,$s2,10      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsra      $gp,$s2,$s3     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsra32    $gp,10          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsra32    $gp,$s2,10      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsrav     $gp,$s2,$s3     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsrl      $s3,23          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsrl      $s3,$6,23       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsrl      $s3,$6,$s4      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsrl32    $s3,23          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsrl32    $s3,$6,23       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsrlv     $s3,$t2,$s4     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsubu     $a1,$a1,$k0     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        eret                      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        floor.l.d $f26,$f7        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        floor.l.s $f12,$f5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ldxc1     $f8,$s7($t3)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        luxc1     $f19,$s6($s5)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lwxc1     $f12,$s1($s8)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf      $gp,$a0,$fcc0   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf      $gp,$a0,$fcc7   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
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
        movz      $a1,$s6,$a1     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.d    $f12,$f29,$a1   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.s    $f25,$f7,$v1    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        pll.ps    $f25,$f9,$f30   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        plu.ps    $f1,$f26,$f29   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        pul.ps    $f9,$f30,$f26     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        puu.ps    $f24,$f9,$f2      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        round.l.d $f12,$f1        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        round.l.s $f25,$f5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        trunc.l.d $f23,$f23       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        trunc.l.s $f28,$f31       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdxc1     $f11,$a2($t2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
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

