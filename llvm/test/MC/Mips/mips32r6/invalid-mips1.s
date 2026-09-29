# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        add       $9,$14,15176        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        add       $24,-7193           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        addi      $13,$9,26322        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ngl.d   $f29,$f29           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ngle.d  $f0,$f16            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.sf.d    $f30,$f0            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.sf.s    $f14,$f22           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mfhi      $s3                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mfhi      $sp                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mflo      $s1                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mthi      $s1                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mtlo      $25                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mtlo      $sp                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mult      $sp,$s4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mult      $sp,$v0             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        multu     $9,$s2              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        multu     $gp,$k0             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
#       div has been re-encoded. See valid.s
#       divu has been re-encoded. See valid.s
        sub       $22,$17,-3126       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sub       $13,6512            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
