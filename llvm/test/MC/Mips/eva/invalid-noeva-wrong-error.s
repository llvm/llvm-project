# invalid operand for instructions that are invalid without -mattr=+eva flag and
# are correctly rejected but use the wrong error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r3 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r5 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r6 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r2 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r3 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r5 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r6 2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        cachee    31, 255($7)          # CHECK: :[[@LINE]]:23: error: invalid operand for instruction
        cachee    0, -256($4)          # CHECK: :[[@LINE]]:22: error: invalid operand for instruction
        cachee    5, -140($4)          # CHECK: :[[@LINE]]:22: error: invalid operand for instruction
        lbe       $10,-256($25)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lbe       $13,255($15)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lbe       $11,146($14)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lbue      $13,-256($v1)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lbue      $13,255($v0)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lbue      $13,-190($v1)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhe       $13,-256($s5)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhe       $12,255($s0)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhe       $13,81($s0)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhue      $s2,-256($v1)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhue      $s2,255($v1)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhue      $s6,-168($v0)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lle       $v0,-256($s5)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lle       $v1,255($s3)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lle       $v1,-71($s6)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwe       $15,255($a2)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwe       $13,-256($a2)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwe       $15,-200($a1)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwle      $s6,255($15)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwle      $s7,-256($10)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwle      $s7,-176($13)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwre      $zero,255($gp)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwre      $zero,-256($gp)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwre      $zero,-176($gp)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        prefe     14, -256($2)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        prefe     11, 255($3)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        prefe     14, -37($3)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sbe       $s1,255($11)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sbe       $s1,-256($10)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sbe       $s3,0($14)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sce       $9,255($s2)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sce       $12,-256($s5)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sce       $13,-31($s7)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        she       $14,255($15)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        she       $14,-256($15)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        she       $9,235($11)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swe       $ra,255($sp)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swe       $ra,-256($sp)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swe       $ra,-53($sp)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swle      $9,255($s1)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swle      $10,-256($s3)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swle      $8,131($s5)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swre      $s4,255($13)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swre      $s4,-256($13)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swre      $s2,86($14)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
