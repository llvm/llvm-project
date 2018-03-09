# Instructions that are invalid as they were removed in R6
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=+eva 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r6 -mattr=+eva 2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        lwle      $s6,255($15)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwle      $s7,-256($10)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwle      $s7,-176($13)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwre      $zero,255($gp)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwre      $zero,-256($gp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwre      $zero,-176($gp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swle      $9,255($s1)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swle      $10,-256($s3)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swle      $8,131($s5)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swre      $s4,255($13)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swre      $s4,-256($13)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swre      $s2,86($14)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lle       $33, 8($5)         # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
        lle       $4, 8($33)         # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        lle       $4, 512($5)        # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        lle       $4, -513($5)       # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        lwe       $33, 8($5)         # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
        lwe       $4, 8($33)         # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        lwe       $4, 512($5)        # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        lwe       $4, -513($5)       # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        sbe       $33, 8($5)         # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
        sbe       $4, 8($33)         # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        sbe       $4, 512($5)        # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        sbe       $4, -513($5)       # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        sce       $33, 8($5)         # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
        sce       $4, 8($33)         # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        sce       $4, 512($5)        # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        sce       $4, -513($5)       # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        she       $33, 8($5)         # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
        she       $4, 8($33)         # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        she       $4, 512($5)        # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        she       $4, -513($5)       # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        swe       $33, 8($4)         # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
        swe       $5, 8($34)         # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        swe       $5, 512($4)        # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
        swe       $5, -513($4)       # CHECK: :[[@LINE]]:23: error: expected memory with 9-bit signed offset
