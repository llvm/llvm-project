# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding \
# RUN:            -mcpu=mips32r2 | FileCheck -check-prefix=O32 %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding \
# RUN:            -mcpu=mips32r6 | FileCheck -check-prefix=O32 %s

# N32 should be acceptable too but it currently errors out.
# N64 should be acceptable too but we cannot convert la to dla yet.

.option pic2
la $5, symbol
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT

la $5, symbol($6)
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $5, $5, $6             # encoding: [0x00,0xa6,0x28,0x21]

la $6, symbol($6)
# O32: lw $1, %got(symbol)($gp)    # encoding: [0x8f,0x81,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $6, $1, $6             # encoding: [0x00,0x26,0x30,0x21]

la $5, symbol+8
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $5, $5, 8             # encoding: [0x24,0xa5,0x00,0x08]

la $5, symbol+8($6)
# O32: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $5, $5, 8             # encoding: [0x24,0xa5,0x00,0x08]
# O32: addu $5, $5, $6             # encoding: [0x00,0xa6,0x28,0x21]

la $6, symbol+8($6)
# O32: lw $1, %got(symbol)($gp)    # encoding: [0x8f,0x81,A,A]
# O32:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $1, $1, 8             # encoding: [0x24,0x21,0x00,0x08]
# O32: addu $6, $1, $6             # encoding: [0x00,0x26,0x30,0x21]

la $5, 1f
# O32: lw $5, %got($tmp0)($gp)     # encoding: [0x8f,0x85,A,A]
# O32:                             #   fixup A - offset: 0, value: %got($tmp0), kind: fixup_Mips_GOT
# O32: addiu $5, $5, %lo($tmp0)    # encoding: [0x24,0xa5,A,A]
# O32:                             #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_Mips_LO16
1:

# PIC expansions involving $25 are special.
la $25, symbol
# O32: lw $25, %call16(symbol)($gp) # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %call16(symbol), kind: fixup_Mips_CALL16

la $25, symbol($6)
# O32: lw $25, %got(symbol)($gp)    # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $25, $25, $6            # encoding: [0x03,0x26,0xc8,0x21]

la $25, symbol($25)
# O32: lw $1, %got(symbol)($gp)     # encoding: [0x8f,0x81,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addu $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x21]

la $25, symbol+8
# O32: lw $25, %got(symbol)($gp)    # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $25, $25, 8            # encoding: [0x27,0x39,0x00,0x08]

la $25, symbol+8($6)
# O32: lw $25, %got(symbol)($gp)    # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $25, $25, 8            # encoding: [0x27,0x39,0x00,0x08]
# O32: addu $25, $25, $6            # encoding: [0x03,0x26,0xc8,0x21]

la $25, symbol+8($25)
# O32: lw $1, %got(symbol)($gp)     # encoding: [0x8f,0x81,A,A]
# O32:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
# O32: addiu $1, $1, 8              # encoding: [0x24,0x21,0x00,0x08]
# O32: addu $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x21]

la $25, 2f
# O32: lw $25, %got($tmp1)($gp)     # encoding: [0x8f,0x99,A,A]
# O32:                              #   fixup A - offset: 0, value: %got($tmp1), kind: fixup_Mips_GOT
# O32: addiu $25, $25, %lo($tmp1)   # encoding: [0x27,0x39,A,A]
# O32:                              #   fixup A - offset: 0, value: %lo($tmp1), kind: fixup_Mips_LO16
2:
