# Instructions that are invalid as they were removed in R6
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=+eva 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r6 -mattr=+eva 2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        lwle      $s6,255($15)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lwle      $s7,-256($10)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lwle      $s7,-176($13)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lwre      $zero,255($gp)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lwre      $zero,-256($gp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lwre      $zero,-176($gp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        swle      $9,255($s1)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        swle      $10,-256($s3)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        swle      $8,131($s5)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        swre      $s4,255($13)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        swre      $s4,-256($13)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        swre      $s2,86($14)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
