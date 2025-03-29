# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding \
# RUN:            -mcpu=mips32r2 | FileCheck -check-prefix=O32 %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding \
# RUN:            -mcpu=mips32r6 | FileCheck -check-prefix=O32 %s

# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding \
# RUN:            -mattr=+xgot -mcpu=mips32r2 \
# RUN:   | FileCheck -check-prefixes=XGOT,XO32 %s

# RUN: llvm-mc %s -triple=mipsn32 -show-encoding \
# RUN:            -mcpu=mips64r2 | FileCheck -check-prefix=N32 %s

# RUN: llvm-mc %s -triple=mipsn32 -show-encoding \
# RUN:            -mattr=+xgot -mcpu=mips64r2 \
# RUN:   | FileCheck -check-prefixes=XGOT,XN32 %s

# N64 should be acceptable too but we cannot convert la to dla yet.

.option pic2
la $5, symbol
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT

# XGOT: lui  $5, %got_hi(symbol)      # encoding: [0x3c,0x05,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: addu $5, $5, $gp              # encoding: [0x00,0xbc,0x28,0x21]
# XGOT: lw   $5, %got_lo(symbol)($5)  # encoding: [0x8c,0xa5,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16

# N32: lw $5, %got_disp(symbol)($gp)  # encoding: [0x8f,0x85,A,A]
                                      #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP

la $5, symbol($6)
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $5, $5, $6             # encoding: [0x00,0xa6,0x28,0x21]

# XGOT: lui  $5, %got_hi(symbol)      # encoding: [0x3c,0x05,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: addu $5, $5, $gp              # encoding: [0x00,0xbc,0x28,0x21]
# XGOT: lw   $5, %got_lo(symbol)($5)  # encoding: [0x8c,0xa5,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addu $5, $5, $6               # encoding: [0x00,0xa6,0x28,0x21]

# N32: lw   $5, %got_disp(symbol)($gp)  # encoding: [0x8f,0x85,A,A]
# N32:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addu $5, $5, $6                  # encoding: [0x00,0xa6,0x28,0x21]

la $6, symbol($6)
# O32: lw $1, %got(symbol)($gp)    # encoding: [0x8f,0x81,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $6, $1, $6             # encoding: [0x00,0x26,0x30,0x21]

# XGOT: lui  $1, %got_hi(symbol)      # encoding: [0x3c,0x01,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: addu $1, $1, $gp              # encoding: [0x00,0x3c,0x08,0x21]
# XGOT: lw   $1, %got_lo(symbol)($1)  # encoding: [0x8c,0x21,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addu $6, $1, $6               # encoding: [0x00,0x26,0x30,0x21]

# N32: lw $1, %got_disp(symbol)($gp)  # encoding: [0x8f,0x81,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addu $6, $1, $6                # encoding: [0x00,0x26,0x30,0x21]

la $5, symbol+8
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $5, $5, 8             # encoding: [0x24,0xa5,0x00,0x08]

# XGOT: lui   $5, %got_hi(symbol+8)   # encoding: [0x3c,0x05,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: addu  $5, $5, $gp             # encoding: [0x00,0xbc,0x28,0x21]
# XGOT: lw    $5, %got_lo(symbol)($5) # encoding: [0x8c,0xa5,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addiu $5, $5, 8               # encoding: [0x24,0xa5,0x00,0x08]

# N32: lw $5, %got_disp(symbol)($gp)  # encoding: [0x8f,0x85,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addiu $5, $5, 8                # encoding: [0x24,0xa5,0x00,0x08]

la $5, symbol+8($6)
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $5, $5, 8             # encoding: [0x24,0xa5,0x00,0x08]
# O32: addu $5, $5, $6             # encoding: [0x00,0xa6,0x28,0x21]

# XGOT: lui   $5, %got_hi(symbol+8)   # encoding: [0x3c,0x05,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: addu  $5, $5, $gp             # encoding: [0x00,0xbc,0x28,0x21]
# XGOT: lw    $5, %got_lo(symbol)($5) # encoding: [0x8c,0xa5,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addiu $5, $5, 8               # encoding: [0x24,0xa5,0x00,0x08]
# XGOT: addu  $5, $5, $6              # encoding: [0x00,0xa6,0x28,0x21]

# N32: lw $5, %got_disp(symbol)($gp)  # encoding: [0x8f,0x85,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addiu $5, $5, 8                # encoding: [0x24,0xa5,0x00,0x08]
# N32: addu $5, $5, $6                # encoding: [0x00,0xa6,0x28,0x21]

la $6, symbol+8($6)
# O32: lw $1, %got(symbol)($gp)    # encoding: [0x8f,0x81,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $1, $1, 8             # encoding: [0x24,0x21,0x00,0x08]
# O32: addu $6, $1, $6             # encoding: [0x00,0x26,0x30,0x21]

# XGOT: lui   $1, %got_hi(symbol+8)   # encoding: [0x3c,0x01,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: addu  $1, $1, $gp             # encoding: [0x00,0x3c,0x08,0x21]
# XGOT: lw    $1, %got_lo(symbol)($1) # encoding: [0x8c,0x21,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addiu $1, $1, 8               # encoding: [0x24,0x21,0x00,0x08]
# XGOT: addu  $6, $1, $6              # encoding: [0x00,0x26,0x30,0x21]

# N32: lw $1, %got_disp(symbol)($gp)  # encoding: [0x8f,0x81,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addiu $1, $1, 8                # encoding: [0x24,0x21,0x00,0x08]
# N32: addu $6, $1, $6                # encoding: [0x00,0x26,0x30,0x21]

la $5, 1f
# O32: lw $5, %got($tmp0)($gp)     # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got($tmp0), kind: fixup_Mips_GOT
# O32: addiu $5, $5, %lo($tmp0)    # encoding: [0x24,0xa5,A,A]
# O32:                             #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_Mips_LO16

# XO32: lw    $5, %got($tmp0)($gp)    # encoding: [0x8f,0x85,A,A]
# XO32:                               #   fixup A - offset: 0, value: %got($tmp0), kind: fixup_Mips_GOT
# XO32: addiu $5, $5, %lo($tmp0)      # encoding: [0x24,0xa5,A,A]
# XO32:                               #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_Mips_LO16

# N32: lw     $5, %got_disp(.Ltmp0)($gp)  # encoding: [0x8f,0x85,A,A]
# N32:                                    #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_Mips_GOT_DISP

## FIXME: GAS generates R_MIPS_GOT_PAGE / R_MIPS_GOT_OFST pair of relocations
## in case of using N32 ABI + XGOT.
# XN32: lw    $5, %got_disp(.Ltmp0)($gp)  # encoding: [0x8f,0x85,A,A]
# XN32:                                   #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_Mips_GOT_DISP
1:

# PIC expansions involving $25 are special.
la $25, symbol
# O32: lw $25, %call16(symbol)($gp) # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %call16(symbol), kind: fixup_Mips_CALL16

# XGOT: lui  $25, %call_hi(symbol)      # encoding: [0x3c,0x19,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %call_hi(symbol), kind: fixup_Mips_CALL_HI16
# XGOT: addu $25, $25, $gp              # encoding: [0x03,0x3c,0xc8,0x21]
# XGOT: lw   $25, %call_lo(symbol)($25) # encoding: [0x8f,0x39,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %call_lo(symbol), kind: fixup_Mips_CALL_LO16

# N32: lw $25, %call16(symbol)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                              #   fixup A - offset: 0, value: %call16(symbol), kind: fixup_Mips_CALL16

la $25, symbol($6)
# O32: lw $25, %got(symbol)($gp)    # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $25, $25, $6            # encoding: [0x03,0x26,0xc8,0x21]

# XGOT: lui  $25, %got_hi(symbol)       # encoding: [0x3c,0x19,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: addu $25, $25, $gp              # encoding: [0x03,0x3c,0xc8,0x21]
# XGOT: lw   $25, %got_lo(symbol)($25)  # encoding: [0x8f,0x39,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addu $25, $25, $6               # encoding: [0x03,0x26,0xc8,0x21]

# N32: lw $25, %got_disp(symbol)($gp)   # encoding: [0x8f,0x99,A,A]
# N32:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addu $25, $25, $6                # encoding: [0x03,0x26,0xc8,0x21]

la $25, symbol($25)
# O32: lw $1, %got(symbol)($gp)     # encoding: [0x8f,0x81,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x21]

# XGOT: lui  $1, %got_hi(symbol)      # encoding: [0x3c,0x01,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: addu $1, $1, $gp              # encoding: [0x00,0x3c,0x08,0x21]
# XGOT: lw   $1, %got_lo(symbol)($1)  # encoding: [0x8c,0x21,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addu $25, $1, $25             # encoding: [0x00,0x39,0xc8,0x21]

# N32: lw $1, %got_disp(symbol)($gp)  # encoding: [0x8f,0x81,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addu $25, $1, $25              # encoding: [0x00,0x39,0xc8,0x21]

la $25, symbol+8
# O32: lw $25, %got(symbol)($gp)    # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $25, $25, 8            # encoding: [0x27,0x39,0x00,0x08]

# XGOT: lui   $25, %got_hi(symbol+8)    # encoding: [0x3c,0x19,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: addu  $25, $25, $gp             # encoding: [0x03,0x3c,0xc8,0x21]
# XGOT: lw    $25, %got_lo(symbol)($25) # encoding: [0x8f,0x39,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addiu $25, $25, 8               # encoding: [0x27,0x39,0x00,0x08]

# N32: lw $25, %got_disp(symbol)($gp)   # encoding: [0x8f,0x99,A,A]
# N32:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addiu $25, $25, 8                # encoding: [0x27,0x39,0x00,0x08]

la $25, symbol+8($6)
# O32: lw $25, %got(symbol)($gp)    # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $25, $25, 8            # encoding: [0x27,0x39,0x00,0x08]
# O32: addu $25, $25, $6            # encoding: [0x03,0x26,0xc8,0x21]

# XGOT: lui   $25, %got_hi(symbol+8)    # encoding: [0x3c,0x19,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: addu  $25, $25, $gp             # encoding: [0x03,0x3c,0xc8,0x21]
# XGOT: lw    $25, %got_lo(symbol)($25) # encoding: [0x8f,0x39,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addiu $25, $25, 8               # encoding: [0x27,0x39,0x00,0x08]
# XGOT: addu  $25, $25, $6              # encoding: [0x03,0x26,0xc8,0x21]

# N32: lw $25, %got_disp(symbol)($gp)   # encoding: [0x8f,0x99,A,A]
# N32:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addiu $25, $25, 8                # encoding: [0x27,0x39,0x00,0x08]
# N32: addu $25, $25, $6                # encoding: [0x03,0x26,0xc8,0x21]

la $25, symbol+8($25)
# O32: lw $1, %got(symbol)($gp)     # encoding: [0x8f,0x81,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $1, $1, 8              # encoding: [0x24,0x21,0x00,0x08]
# O32: addu $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x21]

# XGOT: lui   $1, %got_hi(symbol+8)   # encoding: [0x3c,0x01,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: addu  $1, $1, $gp             # encoding: [0x00,0x3c,0x08,0x21]
# XGOT: lw    $1, %got_lo(symbol)($1) # encoding: [0x8c,0x21,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: addiu $1, $1, 8               # encoding: [0x24,0x21,0x00,0x08]
# XGOT: addu  $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x21]

# N32: lw $1, %got_disp(symbol)($gp)  # encoding: [0x8f,0x81,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N32: addiu $1, $1, 8                # encoding: [0x24,0x21,0x00,0x08]
# N32: addu $25, $1, $25              # encoding: [0x00,0x39,0xc8,0x21]

la $25, 2f
# O32: lw $25, %got($tmp1)($gp)     # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got($tmp1), kind: fixup_Mips_GOT
# O32: addiu $25, $25, %lo($tmp1)   # encoding: [0x27,0x39,A,A]
# O32:                              #   fixup A - offset: 0, value: %lo($tmp1), kind: fixup_Mips_LO16

# XO32: lw    $25, %got($tmp1)($gp) # encoding: [0x8f,0x99,A,A]
# XO32:                             #   fixup A - offset: 0, value: %got($tmp1), kind: fixup_Mips_GOT
# XO32: addiu $25, $25, %lo($tmp1)  # encoding: [0x27,0x39,A,A]
# XO32:                             #   fixup A - offset: 0, value: %lo($tmp1), kind: fixup_Mips_LO16

# N32: lw $25, %got_disp(.Ltmp1)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp(.Ltmp1), kind: fixup_Mips_GOT_DISP

## FIXME: GAS generates R_MIPS_GOT_PAGE / R_MIPS_GOT_OFST pair of relocations
## in case of using N32 ABI + XGOT.
# XN32: lw $25, %got_disp(.Ltmp1)($gp)  # encoding: [0x8f,0x99,A,A]
# XN32:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp1), kind: fixup_Mips_GOT_DISP
2:

la $2,.Lstr
# O32:      lw      $2, %got(.Lstr)($gp)          # encoding: [0x8f,0x82,A,A]
# O32-NEXT:                           #   fixup A - offset: 0, value: %got(.Lstr), kind: fixup_Mips_GOT
# O32-NEXT: addiu   $2, $2, %lo(.Lstr)            # encoding: [0x24,0x42,A,A]
# O32-NEXT:                           #   fixup A - offset: 0, value: %lo(.Lstr), kind: fixup_Mips_LO16

# N32:      lw      $2, %got_disp(.Lstr)($gp)     # encoding: [0x8f,0x82,A,A]
# N32-NEXT:                           #   fixup A - offset: 0, value: %got_disp(.Lstr), kind: fixup_Mips_GOT_DISP

la $2,$str2
# O32:      lw      $2, %got($str2)($gp)          # encoding: [0x8f,0x82,A,A]
# O32-NEXT: #   fixup A - offset: 0, value: %got($str2), kind: fixup_Mips_GOT
# O32-NEXT: addiu   $2, $2, %lo($str2)            # encoding: [0x24,0x42,A,A]
# O32-NEXT: #   fixup A - offset: 0, value: %lo($str2), kind: fixup_Mips_LO16

# N32:      lw      $2, %got_disp($str2)($gp)     # encoding: [0x8f,0x82,A,A]
# N32-NEXT:                           #   fixup A - offset: 0, value: %got_disp($str2), kind: fixup_Mips_GOT_DISP

.rodata
.Lstr: .4byte 0
$str2: .4byte 0
