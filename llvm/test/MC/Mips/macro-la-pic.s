# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 | \
# RUN:   FileCheck %s
# N32 should be acceptable too but it currently errors out.
# N64 should be acceptable too but we cannot convert la to dla yet.

.option pic2
la $5, symbol         # CHECK: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
la $5, symbol($6)     # CHECK: lw $5, %got(symbol)($gp)    # encoding: [0x8f,0x85,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
                      # CHECK: addu $5, $5, $6             # encoding: [0x00,0xa6,0x28,0x21]
la $6, symbol($6)     # CHECK: lw $1, %got(symbol)($gp)    # encoding: [0x8f,0x81,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
                      # CHECK: addu $6, $1, $6             # encoding: [0x00,0x26,0x30,0x21]
la $5, symbol+8       # CHECK: lw $5, %got(symbol+8)($gp)  # encoding: [0x8f,0x85,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %got(symbol+8), kind: fixup_Mips_GOT
la $5, symbol+8($6)   # CHECK: lw $5, %got(symbol+8)($gp)  # encoding: [0x8f,0x85,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %got(symbol+8), kind: fixup_Mips_GOT
                      # CHECK: addu $5, $5, $6             # encoding: [0x00,0xa6,0x28,0x21]
la $6, symbol+8($6)   # CHECK: lw $1, %got(symbol+8)($gp)  # encoding: [0x8f,0x81,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %got(symbol+8), kind: fixup_Mips_GOT
                      # CHECK: addiu $1, $1, 8             # encoding: [0x24,0x21,0x00,0x08]
                      # CHECK: addu $6, $1, $6             # encoding: [0x00,0x26,0x30,0x21]
la $5, 1f             # CHECK: lw $5, %got($tmp0)($gp)     # encoding: [0x8f,0x85,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %got($tmp0), kind: fixup_Mips_GOT
                      # CHECK: addiu $5, $5, %lo($tmp0)    # encoding: [0x24,0xa5,A,A]
                      # CHECK:                             #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_Mips_LO16
1:

# PIC expansions involving $25 are special.
la $25, symbol        # CHECK: lw $25, %call16(symbol)($gp) # encoding: [0x8f,0x99,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %call16(symbol), kind: fixup_Mips_CALL16
la $25, symbol($6)    # CHECK: lw $25, %got(symbol)($gp)    # encoding: [0x8f,0x99,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
                      # CHECK: addu $25, $25, $6            # encoding: [0x03,0x26,0xc8,0x21]
la $25, symbol($25)   # CHECK: lw $1, %got(symbol)($gp)     # encoding: [0x8f,0x81,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %got(symbol), kind: fixup_Mips_GOT
                      # CHECK: addu $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x21]
la $25, symbol+8      # CHECK: lw $25, %got(symbol+8)($gp)  # encoding: [0x8f,0x99,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %got(symbol+8), kind: fixup_Mips_GOT
la $25, symbol+8($6)  # CHECK: lw $25, %got(symbol+8)($gp)  # encoding: [0x8f,0x99,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %got(symbol+8), kind: fixup_Mips_GOT
                      # CHECK: addu $25, $25, $6            # encoding: [0x03,0x26,0xc8,0x21]
la $25, symbol+8($25) # CHECK: lw $1, %got(symbol+8)($gp)   # encoding: [0x8f,0x81,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %got(symbol+8), kind: fixup_Mips_GOT
                      # CHECK: addiu $1, $1, 8              # encoding: [0x24,0x21,0x00,0x08]
                      # CHECK: addu $25, $1, $25            # encoding: [0x00,0x39,0xc8,0x21]
la $25, 1f            # CHECK: lw $25, %got($tmp1)($gp)     # encoding: [0x8f,0x99,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %got($tmp1), kind: fixup_Mips_GOT
                      # CHECK: addiu $25, $25, %lo($tmp1)   # encoding: [0x27,0x39,A,A]
                      # CHECK:                              #   fixup A - offset: 0, value: %lo($tmp1), kind: fixup_Mips_LO16
1:
