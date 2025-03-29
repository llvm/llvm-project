# Instructions that are invalid without -mattr=+eva flag. These were rejected
# correctly but used to emit an incorrect error message.
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
        cachee    31, 255($7)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        cachee    0, -256($4)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        cachee    5, -140($4)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lbe       $10,-256($25)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lbe       $13,255($15)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lbe       $11,146($14)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lbue      $13,-256($v1)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lbue      $13,255($v0)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lbue      $13,-190($v1)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lhe       $13,-256($s5)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lhe       $12,255($s0)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lhe       $13,81($s0)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lhue      $s2,-256($v1)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lhue      $s2,255($v1)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lhue      $s6,-168($v0)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lle       $v0,-256($s5)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lle       $v1,255($s3)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lle       $v1,-71($s6)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwe       $15,255($a2)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwe       $13,-256($a2)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwe       $15,-200($a1)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwle      $s6,255($15)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwle      $s7,-256($10)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwle      $s7,-176($13)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwre      $zero,255($gp)       # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwre      $zero,-256($gp)      # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwre      $zero,-176($gp)      # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        prefe     14, -256($2)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        prefe     11, 255($3)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        prefe     14, -37($3)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sbe       $s1,255($11)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sbe       $s1,-256($10)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sbe       $s3,0($14)           # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sce       $9,255($s2)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sce       $12,-256($s5)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sce       $13,-31($s7)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        she       $14,255($15)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        she       $14,-256($15)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        she       $9,235($11)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swe       $ra,255($sp)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swe       $ra,-256($sp)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swe       $ra,-53($sp)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swle      $9,255($s1)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swle      $10,-256($s3)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swle      $8,131($s5)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swre      $s4,255($13)         # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swre      $s4,-256($13)        # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swre      $s2,86($14)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
