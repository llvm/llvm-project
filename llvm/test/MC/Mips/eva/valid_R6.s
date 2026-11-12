# Instructions that are valid
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=+eva | FileCheck %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r6 -mattr=+eva | FileCheck %s
a:
        .set noat
        cachee    31, 255($7)          # CHECK: cachee 31, 255($7)      # encoding: [0x7c,0xff,0x7f,0x9b]
        cachee    0, -256($4)          # CHECK: cachee 0, -256($4)      # encoding: [0x7c,0x80,0x80,0x1b]
        cachee    5, -140($4)          # CHECK: cachee 5, -140($4)      # encoding: [0x7c,0x85,0xba,0x1b]
        lbe       $10,-256($25)        # CHECK: lbe $10, -256($25)      # encoding: [0x7f,0x2a,0x80,0x2c]
        lbe       $13,255($15)         # CHECK: lbe $13, 255($15)       # encoding: [0x7d,0xed,0x7f,0xac]
        lbe       $11,146($14)         # CHECK: lbe $11, 146($14)       # encoding: [0x7d,0xcb,0x49,0x2c]
        lbue      $13,-256($v1)        # CHECK: lbue $13, -256($3)      # encoding: [0x7c,0x6d,0x80,0x28]
        lbue      $13,255($v0)         # CHECK: lbue $13, 255($2)       # encoding: [0x7c,0x4d,0x7f,0xa8]
        lbue      $13,-190($v1)        # CHECK: lbue $13, -190($3)      # encoding: [0x7c,0x6d,0xa1,0x28]
        lhe       $13,-256($s5)        # CHECK: lhe $13, -256($21)      # encoding: [0x7e,0xad,0x80,0x2d]
        lhe       $12,255($s0)         # CHECK: lhe $12, 255($16)       # encoding: [0x7e,0x0c,0x7f,0xad]
        lhe       $13,81($s0)          # CHECK: lhe $13, 81($16)        # encoding: [0x7e,0x0d,0x28,0xad]
        lhue      $s2,-256($v1)        # CHECK: lhue $18, -256($3)      # encoding: [0x7c,0x72,0x80,0x29]
        lhue      $s2,255($v1)         # CHECK: lhue $18, 255($3)       # encoding: [0x7c,0x72,0x7f,0xa9]
        lhue      $s6,-168($v0)        # CHECK: lhue $22, -168($2)      # encoding: [0x7c,0x56,0xac,0x29]
        lle       $v0,-256($s5)        # CHECK: lle $2, -256($21)       # encoding: [0x7e,0xa2,0x80,0x2e]
        lle       $v1,255($s3)         # CHECK: lle $3, 255($19)        # encoding: [0x7e,0x63,0x7f,0xae]
        lle       $v1,-71($s6)         # CHECK: lle $3, -71($22)        # encoding: [0x7e,0xc3,0xdc,0xae]
        lwe       $15,255($a2)         # CHECK: lwe $15, 255($6)        # encoding: [0x7c,0xcf,0x7f,0xaf]
        lwe       $13,-256($a2)        # CHECK: lwe $13, -256($6)       # encoding: [0x7c,0xcd,0x80,0x2f]
        lwe       $15,-200($a1)        # CHECK: lwe $15, -200($5)       # encoding: [0x7c,0xaf,0x9c,0x2f]
        prefe     14, -256($2)         # CHECK: prefe 14, -256($2)      # encoding: [0x7c,0x4e,0x80,0x23]
        prefe     11, 255($3)          # CHECK: prefe 11, 255($3)       # encoding: [0x7c,0x6b,0x7f,0xa3]
        prefe     14, -37($3)          # CHECK: prefe 14, -37($3)       # encoding: [0x7c,0x6e,0xed,0xa3]
        sbe       $s1,255($11)         # CHECK: sbe $17, 255($11)       # encoding: [0x7d,0x71,0x7f,0x9c]
        sbe       $s1,-256($10)        # CHECK: sbe $17, -256($10)      # encoding: [0x7d,0x51,0x80,0x1c]
        sbe       $s3,0($14)           # CHECK: sbe $19, 0($14)         # encoding: [0x7d,0xd3,0x00,0x1c]
        sce       $9,255($s2)          # CHECK: sce $9, 255($18)        # encoding: [0x7e,0x49,0x7f,0x9e]
        sce       $12,-256($s5)        # CHECK: sce $12, -256($21)      # encoding: [0x7e,0xac,0x80,0x1e]
        sce       $13,-31($s7)         # CHECK: sce $13, -31($23)       # encoding: [0x7e,0xed,0xf0,0x9e]
        she       $14,255($15)         # CHECK: she $14, 255($15)       # encoding: [0x7d,0xee,0x7f,0x9d]
        she       $14,-256($15)        # CHECK: she $14, -256($15)      # encoding: [0x7d,0xee,0x80,0x1d]
        she       $9,235($11)          # CHECK: she $9, 235($11)        # encoding: [0x7d,0x69,0x75,0x9d]
        swe       $ra,255($sp)         # CHECK: swe $ra, 255($sp)       # encoding: [0x7f,0xbf,0x7f,0x9f]
        swe       $ra,-256($sp)        # CHECK: swe $ra, -256($sp)      # encoding: [0x7f,0xbf,0x80,0x1f]
        swe       $ra,-53($sp)         # CHECK: swe $ra, -53($sp)       # encoding: [0x7f,0xbf,0xe5,0x9f]
        tlbinv                         # CHECK: tlbinv                  # encoding: [0x42,0x00,0x00,0x03]
        tlbinvf                        # CHECK: tlbinvf                 # encoding: [0x42,0x00,0x00,0x04]


1:
