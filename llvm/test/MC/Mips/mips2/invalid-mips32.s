# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        bc1f      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        bc1t      $fcc1, 4        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
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
        mfc0      $a2,$14,1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf      $gp,$8,$fcc0    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf      $gp,$8,$fcc7    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        movf.d    $f6,$f11,$fcc0  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.d    $f6,$f11,$fcc5  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        movf.s    $f23,$f5,$fcc0  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movf.s    $f23,$f5,$fcc6  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        movn      $v1,$s1,$s0     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn.d    $f27,$f21,$k0   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movn.s    $f12,$f0,$s7    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt      $zero,$s4,$fcc0 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt      $zero,$s4,$fcc5 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        movt.d    $f0,$f2,$fcc0   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.s    $f30,$f2,$fcc0  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movt.s    $f30,$f2,$fcc1  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        movz      $a1,$s6,$9      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.d    $f12,$f29,$9    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        movz.s    $f25,$f7,$v1    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msub      $s7,$k1         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        msubu     $15,$a1         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mtc0      $9,$29,3        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mul       $s0,$s4,$at     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sync      1               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: s-type must be zero or unspecified for pre-MIPS32 ISAs
