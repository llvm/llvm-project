# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        bc1f      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        bc1t      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        clo       $11,$a1         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        clz       $sp,$gp         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        deret                     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        eret                      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        jr.hb     $4              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        jalr.hb   $4              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        jalr.hb   $4, $5          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        madd      $s6,$13         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        madd      $zero,$9        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        maddu     $s3,$gp         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        maddu     $24,$s2         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mfc0      $4, $5, 1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: selector must be zero for pre-MIPS32 ISAs
        mfc2      $4, $5, 1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: selector must be zero for pre-MIPS32 ISAs
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
        movz      $a1,$s6,$9      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.d    $f12,$f29,$9    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.s    $f25,$f7,$v1    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msub      $s7,$k1         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msubu     $15,$a1         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mtc0      $4, $5, 1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: selector must be zero for pre-MIPS32 ISAs
        mtc2      $4, $5, 1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: selector must be zero for pre-MIPS32 ISAs
        mul       $s0,$s4,$at     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sync      1               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: s-type must be zero or unspecified for pre-MIPS32 ISAs
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
