# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding \
# RUN:            -mcpu=mips3 | FileCheck -check-prefix=N64 %s

# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding \
# RUN:            -mattr=+xgot -mcpu=mips3 | FileCheck -check-prefix=XGOT %s

.option pic2
dla $5, symbol
# N64: ld $5, %got_disp(symbol)($gp)    # encoding: [0xdf,0x85,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP

# XGOT: lui     $5, %got_hi(symbol)     # encoding: [0x3c,0x05,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: daddu   $5, $5, $gp             # encoding: [0x00,0xbc,0x28,0x2d]
# XGOT: ld      $5, %got_lo(symbol)($5) # encoding: [0xdc,0xa5,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16

dla $5, symbol($6)
# N64: ld $5, %got_disp(symbol)($gp)    # encoding: [0xdf,0x85,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddu $5, $5, $6                 # encoding: [0x00,0xa6,0x28,0x2d]

# XGOT: lui     $5, %got_hi(symbol)     # encoding: [0x3c,0x05,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: daddu   $5, $5, $gp             # encoding: [0x00,0xbc,0x28,0x2d]
# XGOT: ld      $5, %got_lo(symbol)($5) # encoding: [0xdc,0xa5,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddu   $5, $5, $6              # encoding: [0x00,0xa6,0x28,0x2d]

dla $6, symbol($6)
# N64: ld $1, %got_disp(symbol)($gp)    # encoding: [0xdf,0x81,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddu $6, $1, $6                 # encoding: [0x00,0x26,0x30,0x2d]

# XGOT: lui     $1, %got_hi(symbol)     # encoding: [0x3c,0x01,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: daddu   $1, $1, $gp             # encoding: [0x00,0x3c,0x08,0x2d]
# XGOT: ld      $1, %got_lo(symbol)($1) # encoding: [0xdc,0x21,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddu   $6, $1, $6              # encoding: [0x00,0x26,0x30,0x2d]

dla $5, symbol+8
# N64: ld $5, %got_disp(symbol)($gp)    # encoding: [0xdf,0x85,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddiu $5, $5, 8                 # encoding: [0x64,0xa5,0x00,0x08]

# XGOT: lui     $5, %got_hi(symbol+8)   # encoding: [0x3c,0x05,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: daddu   $5, $5, $gp             # encoding: [0x00,0xbc,0x28,0x2d]
# XGOT: ld      $5, %got_lo(symbol)($5) # encoding: [0xdc,0xa5,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddiu  $5, $5, 8               # encoding: [0x64,0xa5,0x00,0x08]

dla $5, symbol+8($6)
# N64: ld $5, %got_disp(symbol)($gp)    # encoding: [0xdf,0x85,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddiu $5, $5, 8                 # encoding: [0x64,0xa5,0x00,0x08]
# N64: daddu $5, $5, $6                 # encoding: [0x00,0xa6,0x28,0x2d]

# XGOT: lui     $5, %got_hi(symbol+8)   # encoding: [0x3c,0x05,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: daddu   $5, $5, $gp             # encoding: [0x00,0xbc,0x28,0x2d]
# XGOT: ld      $5, %got_lo(symbol)($5) # encoding: [0xdc,0xa5,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddiu  $5, $5, 8               # encoding: [0x64,0xa5,0x00,0x08]
# XGOT: daddu   $5, $5, $6              # encoding: [0x00,0xa6,0x28,0x2d]

dla $6, symbol+8($6)
# N64: ld $1, %got_disp(symbol)($gp)    # encoding: [0xdf,0x81,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddiu $1, $1, 8                 # encoding: [0x64,0x21,0x00,0x08]
# N64: daddu $6, $1, $6                 # encoding: [0x00,0x26,0x30,0x2d]

# XGOT: lui     $1, %got_hi(symbol+8)   # encoding: [0x3c,0x01,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: daddu   $1, $1, $gp             # encoding: [0x00,0x3c,0x08,0x2d]
# XGOT: ld      $1, %got_lo(symbol)($1) # encoding: [0xdc,0x21,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddiu  $1, $1, 8               # encoding: [0x64,0x21,0x00,0x08]
# XGOT: daddu   $6, $1, $6              # encoding: [0x00,0x26,0x30,0x2d]

dla $5, 1f
# N64: ld $5, %got_disp(.Ltmp0)($gp)    # encoding: [0xdf,0x85,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_Mips_GOT_DISP

## FIXME: GAS generates R_MIPS_GOT_PAGE / R_MIPS_GOT_OFST pair of relocations.
# XGOT: ld $5, %got_disp(.Ltmp0)($gp)   # encoding: [0xdf,0x85,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_Mips_GOT_DISP
1:

# PIC expansions involving $25 are special.
dla $25, symbol
# N64: ld $25, %call16(symbol)($gp)     # encoding: [0xdf,0x99,A,A]
# N64:                                  #   fixup A - offset: 0, value: %call16(symbol), kind: fixup_Mips_CALL16

# XGOT: lui   $25, %call_hi(symbol)       # encoding: [0x3c,0x19,A,A]
# XGOT:                                   #   fixup A - offset: 0, value: %call_hi(symbol), kind: fixup_Mips_CALL_HI16
# XGOT: daddu $25, $25, $gp               # encoding: [0x03,0x3c,0xc8,0x2d]
# XGOT: ld    $25, %call_lo(symbol)($25)  # encoding: [0xdf,0x39,A,A]
# XGOT:                                   #   fixup A - offset: 0, value: %call_lo(symbol), kind: fixup_Mips_CALL_LO16

dla $25, symbol($6)
# N64: ld $25, %got_disp(symbol)($gp)   # encoding: [0xdf,0x99,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddu $25, $25, $6               # encoding: [0x03,0x26,0xc8,0x2d]

# XGOT: lui   $25, %got_hi(symbol)      # encoding: [0x3c,0x19,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: daddu $25, $25, $gp             # encoding: [0x03,0x3c,0xc8,0x2d]
# XGOT: ld    $25, %got_lo(symbol)($25) # encoding: [0xdf,0x39,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddu $25, $25, $6              # encoding: [0x03,0x26,0xc8,0x2d]

dla $25, symbol($25)
# N64: ld $1, %got_disp(symbol)($gp)    # encoding: [0xdf,0x81,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddu $25, $1, $25               # encoding: [0x00,0x39,0xc8,0x2d]

# XGOT: lui   $1, %got_hi(symbol)     # encoding: [0x3c,0x01,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT: daddu $1, $1, $gp             # encoding: [0x00,0x3c,0x08,0x2d]
# XGOT: ld    $1, %got_lo(symbol)($1) # encoding: [0xdc,0x21,A,A]
# XGOT:                               #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddu $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x2d]

dla $25, symbol+8
# N64: ld $25, %got_disp(symbol)($gp)   # encoding: [0xdf,0x99,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddiu $25, $25, 8               # encoding: [0x67,0x39,0x00,0x08]

# XGOT: lui    $25, %got_hi(symbol+8)     # encoding: [0x3c,0x19,A,A]
# XGOT:                                   #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: daddu  $25, $25, $gp              # encoding: [0x03,0x3c,0xc8,0x2d]
# XGOT: ld     $25, %got_lo(symbol)($25)  # encoding: [0xdf,0x39,A,A]
# XGOT:                                   #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddiu $25, $25, 8                # encoding: [0x67,0x39,0x00,0x08]

dla $25, symbol+8($6)
# N64: ld $25, %got_disp(symbol)($gp)   # encoding: [0xdf,0x99,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddiu $25, $25, 8               # encoding: [0x67,0x39,0x00,0x08]
# N64: daddu $25, $25, $6               # encoding: [0x03,0x26,0xc8,0x2d]

# XGOT: lui    $25, %got_hi(symbol+8)     # encoding: [0x3c,0x19,A,A]
# XGOT:                                   #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: daddu  $25, $25, $gp              # encoding: [0x03,0x3c,0xc8,0x2d]
# XGOT: ld     $25, %got_lo(symbol)($25)  # encoding: [0xdf,0x39,A,A]
# XGOT:                                   #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddiu $25, $25, 8                # encoding: [0x67,0x39,0x00,0x08]
# XGOT: daddu  $25, $25, $6               # encoding: [0x03,0x26,0xc8,0x2d]

dla $25, symbol+8($25)
# N64: ld $1, %got_disp(symbol)($gp)    # encoding: [0xdf,0x81,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# N64: daddiu $1, $1, 8                 # encoding: [0x64,0x21,0x00,0x08]
# N64: daddu $25, $1, $25               # encoding: [0x00,0x39,0xc8,0x2d]

# XGOT: lui    $1, %got_hi(symbol+8)    # encoding: [0x3c,0x01,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_hi(symbol+8), kind: fixup_Mips_GOT_HI16
# XGOT: daddu  $1, $1, $gp              # encoding: [0x00,0x3c,0x08,0x2d]
# XGOT: ld     $1, %got_lo(symbol)($1)  # encoding: [0xdc,0x21,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT: daddiu $1, $1, 8                # encoding: [0x64,0x21,0x00,0x08]
# XGOT: daddu  $25, $1, $25             # encoding: [0x00,0x39,0xc8,0x2d]

dla $25, 2f
# N64: ld $25, %got_disp(.Ltmp1)($gp)   # encoding: [0xdf,0x99,A,A]
# N64:                                  #   fixup A - offset: 0, value: %got_disp(.Ltmp1), kind: fixup_Mips_GOT_DISP

## FIXME: GAS generates R_MIPS_GOT_PAGE / R_MIPS_GOT_OFST pair of relocations.
# XGOT: ld $25, %got_disp(.Ltmp1)($gp)  # encoding: [0xdf,0x99,A,A]
# XGOT:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp1), kind: fixup_Mips_GOT_DISP
2:
