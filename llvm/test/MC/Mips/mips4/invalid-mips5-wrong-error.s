# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips4 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        abs.ps    $f22,$f8            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        add.ps    $f25,$f27,$f13      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        alnv.ps   $f12,$f18,$f30,$t0  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.eq.ps   $fcc5,$f0,$f9       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.f.ps    $fcc6,$f11,$f11     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.le.ps   $fcc1,$f7,$f20      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.lt.ps   $f19,$f5            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.nge.ps  $f1,$f26            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.ngl.ps  $f21,$f30           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.ngle.ps $fcc7,$f12,$f20     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.ngt.ps  $fcc5,$f30,$f6      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.ole.ps  $fcc7,$f21,$f8      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.olt.ps  $fcc3,$f7,$f16      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.seq.ps  $fcc6,$f31,$f14     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.sf.ps   $fcc6,$f4,$f6       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.ueq.ps  $fcc1,$f5,$f29      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.ule.ps  $fcc6,$f17,$f3      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.ult.ps  $fcc7,$f14,$f0      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        c.un.ps   $fcc4,$f2,$f26      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        madd.ps   $f22,$f3,$f14,$f3   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        mov.ps    $f22,$f17           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        movf.ps   $f10,$f28,$fcc6     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        movn.ps   $f31,$f31,$s3       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        movt.ps   $f20,$f25,$fcc2     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        movz.ps   $f18,$f17,$ra       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        msub.ps   $f12,$f14,$f29,$f17 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        mul.ps    $f14,$f0,$f16       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        neg.ps    $f19,$f13           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        nmadd.ps  $f27,$f4,$f9,$f25   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        nmsub.ps  $f6,$f12,$f14,$f17  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        sub.ps    $f5,$f14,$f26       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
