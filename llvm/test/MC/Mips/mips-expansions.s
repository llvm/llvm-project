# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-LE
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-BE

# Check that the IAS expands macro instructions in the same way as GAS.

# Load address, done by MipsAsmParser::expandLoadAddressReg()
# and MipsAsmParser::expandLoadAddressImm():
  la $8, 1f
# CHECK-LE: lui     $8, %hi($tmp0)        # encoding: [A,A,0x08,0x3c]
# CHECK-LE:                               #   fixup A - offset: 0, value: %hi($tmp0), kind: fixup_Mips_HI16
# CHECK-LE: addiu   $8, $8, %lo($tmp0)    # encoding: [A,A,0x08,0x25]
# CHECK-LE:                               #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_Mips_LO16

  lb $4, 0x8000
# CHECK-LE: lui     $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# CHECK-LE: lb      $4, -32768($4)          # encoding: [0x00,0x80,0x84,0x80]

  lb  $4, 0x20004($3)
# CHECK-LE: lui     $4, 2                   # encoding: [0x02,0x00,0x04,0x3c]
# CHECK-LE: addu    $4, $4, $3              # encoding: [0x21,0x20,0x83,0x00]
# CHECK-LE: lb      $4, 4($4)               # encoding: [0x04,0x00,0x84,0x80]

  lbu $4, 0x8000
# CHECK-LE: lui     $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# CHECK-LE: lbu     $4, -32768($4)          # encoding: [0x00,0x80,0x84,0x90]

  lbu  $4, 0x20004($3)
# CHECK-LE: lui     $4, 2                   # encoding: [0x02,0x00,0x04,0x3c]
# CHECK-LE: addu    $4, $4, $3              # encoding: [0x21,0x20,0x83,0x00]
# CHECK-LE: lbu     $4, 4($4)               # encoding: [0x04,0x00,0x84,0x90]

# LW/SW and LDC1/SDC1 of symbol address, done by MipsAsmParser::expandMemInst():
  .set noat
  lw $10, symbol($4)
# CHECK-LE: lui     $10, %hi(symbol)        # encoding: [A,A,0x0a,0x3c]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %hi(symbol), kind: fixup_Mips_HI16
# CHECK-LE: addu    $10, $10, $4            # encoding: [0x21,0x50,0x44,0x01]
# CHECK-LE: lw      $10, %lo(symbol)($10)   # encoding: [A,A,0x4a,0x8d]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %lo(symbol), kind: fixup_Mips_LO16
  .set at
  sw $10, symbol($9)
# CHECK-LE: lui     $1, %hi(symbol)         # encoding: [A,A,0x01,0x3c]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %hi(symbol), kind: fixup_Mips_HI16
# CHECK-LE: addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: sw      $10, %lo(symbol)($1)    # encoding: [A,A,0x2a,0xac]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %lo(symbol), kind: fixup_Mips_LO16

  lw $8, 1f
# CHECK-LE: lui $8, %hi($tmp0)              # encoding: [A,A,0x08,0x3c]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %hi($tmp0), kind: fixup_Mips_HI16
# CHECK-LE: lw  $8, %lo($tmp0)($8)          # encoding: [A,A,0x08,0x8d]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_Mips_LO16
  sw $8, 1f
# CHECK-LE: lui $1, %hi($tmp0)              # encoding: [A,A,0x01,0x3c]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %hi($tmp0), kind: fixup_Mips_HI16
# CHECK-LE: sw  $8, %lo($tmp0)($1)          # encoding: [A,A,0x28,0xac]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_Mips_LO16

  lw $10, 655483($4)
# CHECK-LE: lui     $10, 10                 # encoding: [0x0a,0x00,0x0a,0x3c]
# CHECK-LE: addu    $10, $10, $4            # encoding: [0x21,0x50,0x44,0x01]
# CHECK-LE: lw      $10, 123($10)           # encoding: [0x7b,0x00,0x4a,0x8d]
  sw $10, 123456($9)
# CHECK-LE: lui     $1, 2                   # encoding: [0x02,0x00,0x01,0x3c]
# CHECK-LE: addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: sw      $10, -7616($1)          # encoding: [0x40,0xe2,0x2a,0xac]

  lw $8, symbol
# CHECK-LE:     lui     $8, %hi(symbol)     # encoding: [A,A,0x08,0x3c]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %hi(symbol), kind: fixup_Mips_HI16
# CHECK-LE-NOT: move    $8, $8              # encoding: [0x21,0x40,0x00,0x01]
# CHECK-LE:     lw      $8, %lo(symbol)($8) # encoding: [A,A,0x08,0x8d]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %lo(symbol), kind: fixup_Mips_LO16
  sw $8, symbol
# CHECK-LE:     lui     $1, %hi(symbol)     # encoding: [A,A,0x01,0x3c]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %hi(symbol), kind: fixup_Mips_HI16
# CHECK-LE-NOT: move    $1, $1              # encoding: [0x21,0x08,0x20,0x00]
# CHECK-LE:     sw      $8, %lo(symbol)($1) # encoding: [A,A,0x28,0xac]
# CHECK-LE:                                 #   fixup A - offset: 0, value: %lo(symbol), kind: fixup_Mips_LO16

  ldc1 $f0, symbol
# CHECK-LE: lui     $1, %hi(symbol)
# CHECK-LE: ldc1    $f0, %lo(symbol)($1)
  sdc1 $f0, symbol
# CHECK-LE: lui     $1, %hi(symbol)
# CHECK-LE: sdc1    $f0, %lo(symbol)($1)

# Test BNE with an immediate as the 2nd operand.
  bne $2, 0, 1332
# CHECK-LE: bnez  $2, 1332          # encoding: [0x4d,0x01,0x40,0x14]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, 123, 1332
# CHECK-LE: addiu $1, $zero, 123    # encoding: [0x7b,0x00,0x01,0x24]
# CHECK-LE: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, -2345, 1332
# CHECK-LE: addiu $1, $zero, -2345  # encoding: [0xd7,0xf6,0x01,0x24]
# CHECK-LE: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, 65538, 1332
# CHECK-LE: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori   $1, $1, 2         # encoding: [0x02,0x00,0x21,0x34]
# CHECK-LE: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, ~7, 1332
# CHECK-LE: addiu $1, $zero, -8     # encoding: [0xf8,0xff,0x01,0x24]
# CHECK-LE: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, 0x10000, 1332
# CHECK-LE: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

# Test BEQ with an immediate as the 2nd operand.
  beq $2, 0, 1332
# CHECK-LE: beqz  $2, 1332          # encoding: [0x4d,0x01,0x40,0x10]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 123, 1332
# CHECK-LE: addiu $1, $zero, 123    # encoding: [0x7b,0x00,0x01,0x24]
# CHECK-LE: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, -2345, 1332
# CHECK-LE: addiu $1, $zero, -2345  # encoding: [0xd7,0xf6,0x01,0x24]
# CHECK-LE: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 65538, 1332
# CHECK-LE: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori   $1, $1, 2         # encoding: [0x02,0x00,0x21,0x34]
# CHECK-LE: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, ~7, 1332
# CHECK-LE: addiu $1, $zero, -8     # encoding: [0xf8,0xff,0x01,0x24]
# CHECK-LE: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 0x10000, 1332
# CHECK-LE: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 65538, foo
# CHECK-LE: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori   $1, $1, 2         # encoding: [0x02,0x00,0x21,0x34]
# CHECK-LE: beq   $2, $1, foo       # encoding: [A,A,0x41,0x10]
# CHECK-LE: nop                     # encoding: [0x00,0x00,0x00,0x00]

# Test ULH with immediate operand.
ulh_imm: # CHECK-LABEL: ulh_imm:
  ulh $8, 0
# CHECK-BE: lb   $1, 0($zero)      # encoding: [0x80,0x01,0x00,0x00]
# CHECK-BE: lbu  $8, 1($zero)      # encoding: [0x90,0x08,0x00,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lb   $1, 1($zero)      # encoding: [0x01,0x00,0x01,0x80]
# CHECK-LE: lbu  $8, 0($zero)      # encoding: [0x00,0x00,0x08,0x90]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 2
# CHECK-BE: lb   $1, 2($zero)      # encoding: [0x80,0x01,0x00,0x02]
# CHECK-BE: lbu  $8, 3($zero)      # encoding: [0x90,0x08,0x00,0x03]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lb   $1, 3($zero)      # encoding: [0x03,0x00,0x01,0x80]
# CHECK-LE: lbu  $8, 2($zero)      # encoding: [0x02,0x00,0x08,0x90]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 0x8000
# CHECK-BE: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: ori  $1, $zero, 32768  # encoding: [0x00,0x80,0x01,0x34]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, -0x8000
# CHECK-BE: lb   $1, -32768($zero) # encoding: [0x80,0x01,0x80,0x00]
# CHECK-BE: lbu  $8, -32767($zero) # encoding: [0x90,0x08,0x80,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lb   $1, -32767($zero) # encoding: [0x01,0x80,0x01,0x80]
# CHECK-LE: lbu  $8, -32768($zero) # encoding: [0x00,0x80,0x08,0x90]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 0x10000
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 0x18888
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: ori  $1, $1, 34952     # encoding: [0x34,0x21,0x88,0x88]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 34952     # encoding: [0x88,0x88,0x21,0x34]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, -32769
# CHECK-BE: lui  $1, 65535         # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-BE: ori  $1, $1, 32767     # encoding: [0x34,0x21,0x7f,0xff]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 65535         # encoding: [0xff,0xff,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 32767     # encoding: [0xff,0x7f,0x21,0x34]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 32767
# CHECK-BE: addiu $1, $zero, 32767  # encoding: [0x24,0x01,0x7f,0xff]
# CHECK-BE: lb   $8, 0($1)          # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)          # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8          # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1         # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: addiu $1, $zero, 32767  # encoding: [0xff,0x7f,0x01,0x24]
# CHECK-LE: lb   $8, 1($1)          # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)          # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8          # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1         # encoding: [0x25,0x40,0x01,0x01]

# Test ULH with immediate offset and a source register operand.
ulh_reg: # CHECK-LABEL: ulh_reg:
  ulh $8, 0($9)
# CHECK-BE: lb   $1, 0($9)         # encoding: [0x81,0x21,0x00,0x00]
# CHECK-BE: lbu  $8, 1($9)         # encoding: [0x91,0x28,0x00,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lb   $1, 1($9)         # encoding: [0x01,0x00,0x21,0x81]
# CHECK-LE: lbu  $8, 0($9)         # encoding: [0x00,0x00,0x28,0x91]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 2($9)
# CHECK-BE: lb   $1, 2($9)         # encoding: [0x81,0x21,0x00,0x02]
# CHECK-BE: lbu  $8, 3($9)         # encoding: [0x91,0x28,0x00,0x03]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lb   $1, 3($9)         # encoding: [0x03,0x00,0x21,0x81]
# CHECK-LE: lbu  $8, 2($9)         # encoding: [0x02,0x00,0x28,0x91]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 0x8000($9)
# CHECK-BE: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: ori  $1, $zero, 32768  # encoding: [0x00,0x80,0x01,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, -0x8000($9)
# CHECK-BE: lb   $1, -32768($9)    # encoding: [0x81,0x21,0x80,0x00]
# CHECK-BE: lbu  $8, -32767($9)    # encoding: [0x91,0x28,0x80,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lb   $1, -32767($9)    # encoding: [0x01,0x80,0x21,0x81]
# CHECK-LE: lbu  $8, -32768($9)    # encoding: [0x00,0x80,0x28,0x91]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 0x10000($9)
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 0x18888($9)
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: ori  $1, $1, 34952     # encoding: [0x34,0x21,0x88,0x88]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 34952     # encoding: [0x88,0x88,0x21,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, -32769($9)
# CHECK-BE: lui  $1, 65535         # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-BE: ori  $1, $1, 32767     # encoding: [0x34,0x21,0x7f,0xff]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 65535         # encoding: [0xff,0xff,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 32767     # encoding: [0xff,0x7f,0x21,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulh $8, 32767($9)
# CHECK-BE: addiu $1, $9, 32767    # encoding: [0x25,0x21,0x7f,0xff]
# CHECK-BE: lb   $8, 0($1)         # encoding: [0x80,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: addiu $1, $9, 32767    # encoding: [0xff,0x7f,0x21,0x25]
# CHECK-LE: lb   $8, 1($1)         # encoding: [0x01,0x00,0x28,0x80]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

# Test ULHU with immediate operand.
ulhu_imm: # CHECK-LABEL: ulhu_imm:
  ulhu $8, 0
# CHECK-BE: lbu  $1, 0($zero)      # encoding: [0x90,0x01,0x00,0x00]
# CHECK-BE: lbu  $8, 1($zero)      # encoding: [0x90,0x08,0x00,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lbu  $1, 1($zero)      # encoding: [0x01,0x00,0x01,0x90]
# CHECK-LE: lbu  $8, 0($zero)      # encoding: [0x00,0x00,0x08,0x90]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 2
# CHECK-BE: lbu  $1, 2($zero)      # encoding: [0x90,0x01,0x00,0x02]
# CHECK-BE: lbu  $8, 3($zero)      # encoding: [0x90,0x08,0x00,0x03]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lbu  $1, 3($zero)      # encoding: [0x03,0x00,0x01,0x90]
# CHECK-LE: lbu  $8, 2($zero)      # encoding: [0x02,0x00,0x08,0x90]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x8000
# CHECK-BE: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: ori  $1, $zero, 32768  # encoding: [0x00,0x80,0x01,0x34]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -0x8000
# CHECK-BE: lbu  $1, -32768($zero) # encoding: [0x90,0x01,0x80,0x00]
# CHECK-BE: lbu  $8, -32767($zero) # encoding: [0x90,0x08,0x80,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lbu  $1, -32767($zero) # encoding: [0x01,0x80,0x01,0x90]
# CHECK-LE: lbu  $8, -32768($zero) # encoding: [0x00,0x80,0x08,0x90]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x10000
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x18888
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: ori  $1, $1, 34952     # encoding: [0x34,0x21,0x88,0x88]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 34952     # encoding: [0x88,0x88,0x21,0x34]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -32769
# CHECK-BE: lui  $1, 65535         # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-BE: ori  $1, $1, 32767     # encoding: [0x34,0x21,0x7f,0xff]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 65535         # encoding: [0xff,0xff,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 32767     # encoding: [0xff,0x7f,0x21,0x34]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 32767
# CHECK-BE: addiu $1, $zero, 32767  # encoding: [0x24,0x01,0x7f,0xff]
# CHECK-BE: lbu  $8, 0($1)          # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)          # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8          # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1         # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: addiu $1, $zero, 32767  # encoding: [0xff,0x7f,0x01,0x24]
# CHECK-LE: lbu  $8, 1($1)          # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)          # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8          # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1         # encoding: [0x25,0x40,0x01,0x01]

# Test ULHU with immediate offset and a source register operand.
  ulhu $8, 0($9)
# CHECK-BE: lbu  $1, 0($9)         # encoding: [0x91,0x21,0x00,0x00]
# CHECK-BE: lbu  $8, 1($9)         # encoding: [0x91,0x28,0x00,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lbu  $1, 1($9)         # encoding: [0x01,0x00,0x21,0x91]
# CHECK-LE: lbu  $8, 0($9)         # encoding: [0x00,0x00,0x28,0x91]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 2($9)
# CHECK-BE: lbu  $1, 2($9)         # encoding: [0x91,0x21,0x00,0x02]
# CHECK-BE: lbu  $8, 3($9)         # encoding: [0x91,0x28,0x00,0x03]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lbu  $1, 3($9)         # encoding: [0x03,0x00,0x21,0x91]
# CHECK-LE: lbu  $8, 2($9)         # encoding: [0x02,0x00,0x28,0x91]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x8000($9)
# CHECK-BE: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: ori  $1, $zero, 32768  # encoding: [0x00,0x80,0x01,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -0x8000($9)
# CHECK-BE: lbu  $1, -32768($9)    # encoding: [0x91,0x21,0x80,0x00]
# CHECK-BE: lbu  $8, -32767($9)    # encoding: [0x91,0x28,0x80,0x01]
# CHECK-BE: sll  $1, $1, 8         # encoding: [0x00,0x01,0x0a,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lbu  $1, -32767($9)    # encoding: [0x01,0x80,0x21,0x91]
# CHECK-LE: lbu  $8, -32768($9)    # encoding: [0x00,0x80,0x28,0x91]
# CHECK-LE: sll  $1, $1, 8         # encoding: [0x00,0x0a,0x01,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x10000($9)
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x18888($9)
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: ori  $1, $1, 34952     # encoding: [0x34,0x21,0x88,0x88]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 34952     # encoding: [0x88,0x88,0x21,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -32769($9)
# CHECK-BE: lui  $1, 65535         # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-BE: ori  $1, $1, 32767     # encoding: [0x34,0x21,0x7f,0xff]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: lui  $1, 65535         # encoding: [0xff,0xff,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 32767     # encoding: [0xff,0x7f,0x21,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 32767($9)
# CHECK-BE: addiu   $1, $9, 32767  # encoding: [0x25,0x21,0x7f,0xff]
# CHECK-BE: lbu  $8, 0($1)         # encoding: [0x90,0x28,0x00,0x00]
# CHECK-BE: lbu  $1, 1($1)         # encoding: [0x90,0x21,0x00,0x01]
# CHECK-BE: sll  $8, $8, 8         # encoding: [0x00,0x08,0x42,0x00]
# CHECK-BE: or   $8, $8, $1        # encoding: [0x01,0x01,0x40,0x25]
# CHECK-LE: addiu $1, $9, 32767    # encoding: [0xff,0x7f,0x21,0x25]
# CHECK-LE: lbu  $8, 1($1)         # encoding: [0x01,0x00,0x28,0x90]
# CHECK-LE: lbu  $1, 0($1)         # encoding: [0x00,0x00,0x21,0x90]
# CHECK-LE: sll  $8, $8, 8         # encoding: [0x00,0x42,0x08,0x00]
# CHECK-LE: or   $8, $8, $1        # encoding: [0x25,0x40,0x01,0x01]

# Test ULW with immediate operand.
  ulw $8, 0
# CHECK-BE: lwl  $8, 0($zero)      # encoding: [0x88,0x08,0x00,0x00]
# CHECK-BE: lwr  $8, 3($zero)      # encoding: [0x98,0x08,0x00,0x03]
# CHECK-LE: lwl $8, 3($zero)       # encoding: [0x03,0x00,0x08,0x88]
# CHECK-LE: lwr $8, 0($zero)       # encoding: [0x00,0x00,0x08,0x98]

  ulw $8, 2
# CHECK-BE: lwl  $8, 2($zero)      # encoding: [0x88,0x08,0x00,0x02]
# CHECK-BE: lwr  $8, 5($zero)      # encoding: [0x98,0x08,0x00,0x05]
# CHECK-LE: lwl $8, 5($zero)       # encoding: [0x05,0x00,0x08,0x88]
# CHECK-LE: lwr $8, 2($zero)       # encoding: [0x02,0x00,0x08,0x98]

  ulw $8, 0x8000
# CHECK-BE: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: ori $1, $zero, 32768   # encoding: [0x00,0x80,0x01,0x34]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -0x8000
# CHECK-BE: lwl  $8, -32768($zero) # encoding: [0x88,0x08,0x80,0x00]
# CHECK-BE: lwr  $8, -32765($zero) # encoding: [0x98,0x08,0x80,0x03]
# CHECK-LE: lwl $8, -32765($zero)  # encoding: [0x03,0x80,0x08,0x88]
# CHECK-LE: lwr $8, -32768($zero)  # encoding: [0x00,0x80,0x08,0x98]

  ulw $8, 0x10000
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui $1, 1              # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 0x18888
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: ori  $1, $1, 34952     # encoding: [0x34,0x21,0x88,0x88]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui $1, 1              # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori $1, $1, 34952      # encoding: [0x88,0x88,0x21,0x34]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -32771
# CHECK-BE: lui  $1, 65535         # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-BE: ori  $1, $1, 32765     # encoding: [0x34,0x21,0x7f,0xfd]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui $1, 65535          # encoding: [0xff,0xff,0x01,0x3c]
# CHECK-LE: ori $1, $1, 32765      # encoding: [0xfd,0x7f,0x21,0x34]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 32765
# CHECK-BE: addiu $1, $zero, 32765 # encoding: [0x24,0x01,0x7f,0xfd]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: addiu $1, $zero, 32765 # encoding: [0xfd,0x7f,0x01,0x24]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

# Test ULW with immediate offset and a source register operand.
  ulw $8, 0($9)
# CHECK-BE: lwl  $8, 0($9)         # encoding: [0x89,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($9)         # encoding: [0x99,0x28,0x00,0x03]
# CHECK-LE: lwl  $8, 3($9)         # encoding: [0x03,0x00,0x28,0x89]
# CHECK-LE: lwr  $8, 0($9)         # encoding: [0x00,0x00,0x28,0x99]

  ulw $8, 2($9)
# CHECK-BE: lwl  $8, 2($9)         # encoding: [0x89,0x28,0x00,0x02]
# CHECK-BE: lwr  $8, 5($9)         # encoding: [0x99,0x28,0x00,0x05]
# CHECK-LE: lwl  $8, 5($9)         # encoding: [0x05,0x00,0x28,0x89]
# CHECK-LE: lwr  $8, 2($9)         # encoding: [0x02,0x00,0x28,0x99]

  ulw $8, 0x8000($9)
# CHECK-BE: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: ori  $1, $zero, 32768  # encoding: [0x00,0x80,0x01,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lwl  $8, 3($1)         # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr  $8, 0($1)         # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -0x8000($9)
# CHECK-BE: lwl  $8, -32768($9)    # encoding: [0x89,0x28,0x80,0x00]
# CHECK-BE: lwr  $8, -32765($9)    # encoding: [0x99,0x28,0x80,0x03]
# CHECK-LE: lwl  $8, -32765($9)    # encoding: [0x03,0x80,0x28,0x89]
# CHECK-LE: lwr  $8, -32768($9)    # encoding: [0x00,0x80,0x28,0x99]

  ulw $8, 0x10000($9)
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lwl  $8, 3($1)         # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr  $8, 0($1)         # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 0x18888($9)
# CHECK-BE: lui  $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: ori  $1, $1, 34952     # encoding: [0x34,0x21,0x88,0x88]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui  $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 34952     # encoding: [0x88,0x88,0x21,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lwl  $8, 3($1)         # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr  $8, 0($1)         # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -32771($9)
# CHECK-BE: lui  $1, 65535         # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-BE: ori  $1, $1, 32765     # encoding: [0x34,0x21,0x7f,0xfd]
# CHECK-BE: addu $1, $1, $9        # encoding: [0x00,0x29,0x08,0x21]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui  $1, 65535         # encoding: [0xff,0xff,0x01,0x3c]
# CHECK-LE: ori  $1, $1, 32765     # encoding: [0xfd,0x7f,0x21,0x34]
# CHECK-LE: addu $1, $1, $9        # encoding: [0x21,0x08,0x29,0x00]
# CHECK-LE: lwl  $8, 3($1)         # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr  $8, 0($1)         # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 32765($9)
# CHECK-BE: addiu $1, $9, 32765    # encoding: [0x25,0x21,0x7f,0xfd]
# CHECK-BE: lwl  $8, 0($1)         # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr  $8, 3($1)         # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: addiu $1, $9, 32765    # encoding: [0xfd,0x7f,0x21,0x25]
# CHECK-LE: lwl  $8, 3($1)         # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr  $8, 0($1)         # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 0($8)
# CHECK-BE: lwl $1, 0($8)          # encoding: [0x89,0x01,0x00,0x00]
# CHECK-BE: lwr $1, 3($8)          # encoding: [0x99,0x01,0x00,0x03]
# CHECK-BE: move $8, $1            # encoding: [0x00,0x20,0x40,0x25]
# CHECK-LE: lwl $1, 3($8)          # encoding: [0x03,0x00,0x01,0x89]
# CHECK-LE: lwr $1, 0($8)          # encoding: [0x00,0x00,0x01,0x99]
# CHECK-LE: move $8, $1            # encoding: [0x25,0x40,0x20,0x00]

  ulw $8, 2($8)
# CHECK-BE: lwl $1, 2($8)          # encoding: [0x89,0x01,0x00,0x02]
# CHECK-BE: lwr $1, 5($8)          # encoding: [0x99,0x01,0x00,0x05]
# CHECK-BE: move $8, $1            # encoding: [0x00,0x20,0x40,0x25]
# CHECK-LE: lwl $1, 5($8)          # encoding: [0x05,0x00,0x01,0x89]
# CHECK-LE: lwr $1, 2($8)          # encoding: [0x02,0x00,0x01,0x99]
# CHECK-LE: move $8, $1            # encoding: [0x25,0x40,0x20,0x00]

  ulw $8, 0x8000($8)
# CHECK-BE: ori $1, $zero, 32768   # encoding: [0x34,0x01,0x80,0x00]
# CHECK-BE: addu $1, $1, $8        # encoding: [0x00,0x28,0x08,0x21]
# CHECK-BE: lwl $8, 0($1)          # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr $8, 3($1)          # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: ori $1, $zero, 32768   # encoding: [0x00,0x80,0x01,0x34]
# CHECK-LE: addu $1, $1, $8        # encoding: [0x21,0x08,0x28,0x00]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -0x8000($8)
# CHECK-BE: lwl $1, -32768($8)     # encoding: [0x89,0x01,0x80,0x00]
# CHECK-BE: lwr $1, -32765($8)     # encoding: [0x99,0x01,0x80,0x03]
# CHECK-BE: move $8, $1            # encoding: [0x00,0x20,0x40,0x25]
# CHECK-LE: lwl $1, -32765($8)     # encoding: [0x03,0x80,0x01,0x89]
# CHECK-LE: lwr $1, -32768($8)     # encoding: [0x00,0x80,0x01,0x99]
# CHECK-LE: move $8, $1            # encoding: [0x25,0x40,0x20,0x00]

  ulw $8, 0x10000($8)
# CHECK-BE: lui $1, 1              # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: addu $1, $1, $8        # encoding: [0x00,0x28,0x08,0x21]
# CHECK-BE: lwl $8, 0($1)          # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr $8, 3($1)          # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui $1, 1              # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: addu $1, $1, $8        # encoding: [0x21,0x08,0x28,0x00]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 0x18888($8)
# CHECK-BE: lui $1, 1              # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-BE: ori $1, $1, 34952      # encoding: [0x34,0x21,0x88,0x88]
# CHECK-BE: addu $1, $1, $8        # encoding: [0x00,0x28,0x08,0x21]
# CHECK-BE: lwl $8, 0($1)          # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr $8, 3($1)          # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui $1, 1              # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-LE: ori $1, $1, 34952      # encoding: [0x88,0x88,0x21,0x34]
# CHECK-LE: addu $1, $1, $8        # encoding: [0x21,0x08,0x28,0x00]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -32771($8)
# CHECK-BE: lui $1, 65535          # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-BE: ori $1, $1, 32765      # encoding: [0x34,0x21,0x7f,0xfd]
# CHECK-BE: addu $1, $1, $8        # encoding: [0x00,0x28,0x08,0x21]
# CHECK-BE: lwl $8, 0($1)          # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr $8, 3($1)          # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: lui $1, 65535          # encoding: [0xff,0xff,0x01,0x3c]
# CHECK-LE: ori $1, $1, 32765      # encoding: [0xfd,0x7f,0x21,0x34]
# CHECK-LE: addu $1, $1, $8        # encoding: [0x21,0x08,0x28,0x00]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 32765($8)
# CHECK-BE: addiu $1, $8, 32765    # encoding: [0x25,0x01,0x7f,0xfd]
# CHECK-BE: lwl $8, 0($1)          # encoding: [0x88,0x28,0x00,0x00]
# CHECK-BE: lwr $8, 3($1)          # encoding: [0x98,0x28,0x00,0x03]
# CHECK-LE: addiu $1, $8, 32765    # encoding: [0xfd,0x7f,0x01,0x25]
# CHECK-LE: lwl $8, 3($1)          # encoding: [0x03,0x00,0x28,0x88]
# CHECK-LE: lwr $8, 0($1)          # encoding: [0x00,0x00,0x28,0x98]

ush_imm: # CHECK-LABEL: ush_imm
  ush $8, 0
  # CHECK-BE: sb        $8, 1($zero)            # encoding: [0xa0,0x08,0x00,0x01]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, 0($zero)            # encoding: [0xa0,0x01,0x00,0x00]
  # CHECK-LE: sb        $8, 0($zero)            # encoding: [0x00,0x00,0x08,0xa0]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, 1($zero)            # encoding: [0x01,0x00,0x01,0xa0]

  ush $8, 2
  # CHECK-BE: sb        $8, 3($zero)            # encoding: [0xa0,0x08,0x00,0x03]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, 2($zero)            # encoding: [0xa0,0x01,0x00,0x02]
  # CHECK-LE: sb        $8, 2($zero)            # encoding: [0x02,0x00,0x08,0xa0]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, 3($zero)            # encoding: [0x03,0x00,0x01,0xa0]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  ush $8, 0x8000
  # CHECK-BE: ori       $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
  # CHECK-BE: move      $1, $1                  # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: ori       $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
  # CHECK-LE: move      $1, $1                  # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, -0x8000
  # CHECK-BE: sb        $8, -32767($zero)       # encoding: [0xa0,0x08,0x80,0x01]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, -32768($zero)       # encoding: [0xa0,0x01,0x80,0x00]
  # CHECK-LE: sb        $8, -32768($zero)       # encoding: [0x00,0x80,0x08,0xa0]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, -32767($zero)       # encoding: [0x01,0x80,0x01,0xa0]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  ush $8, 0x10000
  # CHECK-BE: lui       $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE: move      $1, $1                  # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE: move      $1, $1                  # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  ush $8, 0x18888
  # CHECK-BE: lui       $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE: ori       $1, $1, 34952           # encoding: [0x34,0x21,0x88,0x88]
  # CHECK-BE: move      $1, $1                  # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE: ori       $1, $1, 34952           # encoding: [0x88,0x88,0x21,0x34]
  # CHECK-LE: move      $1, $1                  # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  ush $8, -32769
  # CHECK-BE: lui       $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
  # CHECK-BE: ori       $1, $1, 32767           # encoding: [0x34,0x21,0x7f,0xff]
  # CHECK-BE: move      $1, $1                  # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
  # CHECK-LE: ori       $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
  # CHECK-LE: move      $1, $1                  # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, 32767
  # CHECK-BE: addiu     $1, $zero, 32767        # encoding: [0x24,0x01,0x7f,0xff]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: addiu     $1, $zero, 32767        # encoding: [0xff,0x7f,0x01,0x24]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

ush_reg: # CHECK-LABEL: ush_reg
  ush $8, 0($9)
  # CHECK-BE: sb        $8, 1($9)               # encoding: [0xa1,0x28,0x00,0x01]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, 0($9)               # encoding: [0xa1,0x21,0x00,0x00]
  # CHECK-LE: sb        $8, 0($9)               # encoding: [0x00,0x00,0x28,0xa1]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, 1($9)               # encoding: [0x01,0x00,0x21,0xa1]

  ush $8, 2($9)
  # CHECK-BE: sb        $8, 3($9)               # encoding: [0xa1,0x28,0x00,0x03]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, 2($9)               # encoding: [0xa1,0x21,0x00,0x02]
  # CHECK-LE: sb        $8, 2($9)               # encoding: [0x02,0x00,0x28,0xa1]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, 3($9)               # encoding: [0x03,0x00,0x21,0xa1]

  ush $8, 0x8000($9)
  # CHECK-BE: ori       $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
  # CHECK-BE: addu      $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: ori       $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
  # CHECK-LE: addu      $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, -0x8000($9)
  # CHECK-BE: sb        $8, -32767($9)          # encoding: [0xa1,0x28,0x80,0x01]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, -32768($9)          # encoding: [0xa1,0x21,0x80,0x00]
  # CHECK-LE: sb        $8, -32768($9)          # encoding: [0x00,0x80,0x28,0xa1]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, -32767($9)          # encoding: [0x01,0x80,0x21,0xa1]

  ush $8, 0x10000($9)
  # CHECK-BE: lui       $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE: addu      $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE: addu      $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, 0x18888($9)
  # CHECK-BE: lui       $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE: ori       $1, $1, 34952           # encoding: [0x34,0x21,0x88,0x88]
  # CHECK-BE: addu      $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE: ori       $1, $1, 34952           # encoding: [0x88,0x88,0x21,0x34]
  # CHECK-LE: addu      $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, -32769($9)
  # CHECK-BE: lui       $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
  # CHECK-BE: ori       $1, $1, 32767           # encoding: [0x34,0x21,0x7f,0xff]
  # CHECK-BE: addu      $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
  # CHECK-LE: ori       $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
  # CHECK-LE: addu      $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, 32767($9)
  # CHECK-BE: addiu     $1, $9, 32767           # encoding: [0x25,0x21,0x7f,0xff]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: addiu     $1, $9, 32767           # encoding: [0xff,0x7f,0x21,0x25]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, 0($8)
  # CHECK-BE: sb        $8, 1($8)               # encoding: [0xa1,0x08,0x00,0x01]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, 0($8)               # encoding: [0xa1,0x01,0x00,0x00]
  # CHECK-LE: sb        $8, 0($8)               # encoding: [0x00,0x00,0x08,0xa1]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, 1($8)               # encoding: [0x01,0x00,0x01,0xa1]

  ush $8, 2($8)
  # CHECK-BE: sb        $8, 3($8)               # encoding: [0xa1,0x08,0x00,0x03]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, 2($8)               # encoding: [0xa1,0x01,0x00,0x02]
  # CHECK-LE: sb        $8, 2($8)               # encoding: [0x02,0x00,0x08,0xa1]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, 3($8)               # encoding: [0x03,0x00,0x01,0xa1]

  ush $8, 0x8000($8)
  # CHECK-BE: ori       $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
  # CHECK-BE: addu      $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: ori       $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
  # CHECK-LE: addu      $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, -0x8000($8)
  # CHECK-BE: sb        $8, -32767($8)          # encoding: [0xa1,0x08,0x80,0x01]
  # CHECK-BE: srl       $1, $8, 8               # encoding: [0x00,0x08,0x0a,0x02]
  # CHECK-BE: sb        $1, -32768($8)          # encoding: [0xa1,0x01,0x80,0x00]
  # CHECK-LE: sb        $8, -32768($8)          # encoding: [0x00,0x80,0x08,0xa1]
  # CHECK-LE: srl       $1, $8, 8               # encoding: [0x02,0x0a,0x08,0x00]
  # CHECK-LE: sb        $1, -32767($8)          # encoding: [0x01,0x80,0x01,0xa1]

  ush $8, 0x10000($8)
  # CHECK-BE: lui       $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE: addu      $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE: addu      $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, 0x18888($8)
  # CHECK-BE: lui       $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE: ori       $1, $1, 34952           # encoding: [0x34,0x21,0x88,0x88]
  # CHECK-BE: addu      $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE: ori       $1, $1, 34952           # encoding: [0x88,0x88,0x21,0x34]
  # CHECK-LE: addu      $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, -32769($8)
  # CHECK-BE: lui       $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
  # CHECK-BE: ori       $1, $1, 32767           # encoding: [0x34,0x21,0x7f,0xff]
  # CHECK-BE: addu      $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: lui       $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
  # CHECK-LE: ori       $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
  # CHECK-LE: addu      $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

  ush $8, 32767($8)
  # CHECK-BE: addiu     $1, $8, 32767           # encoding: [0x25,0x01,0x7f,0xff]
  # CHECK-BE: sb        $8, 1($1)               # encoding: [0xa0,0x28,0x00,0x01]
  # CHECK-BE: srl       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x02]
  # CHECK-BE: sb        $8, 0($1)               # encoding: [0xa0,0x28,0x00,0x00]
  # CHECK-BE: lbu       $1, 0($1)               # encoding: [0x90,0x21,0x00,0x00]
  # CHECK-BE: sll       $8, $8, 8               # encoding: [0x00,0x08,0x42,0x00]
  # CHECK-BE: or        $8, $8, $1              # encoding: [0x01,0x01,0x40,0x25]
  # CHECK-LE: addiu     $1, $8, 32767           # encoding: [0xff,0x7f,0x01,0x25]
  # CHECK-LE: sb        $8, 0($1)               # encoding: [0x00,0x00,0x28,0xa0]
  # CHECK-LE: srl       $8, $8, 8               # encoding: [0x02,0x42,0x08,0x00]
  # CHECK-LE: sb        $8, 1($1)               # encoding: [0x01,0x00,0x28,0xa0]
  # CHECK-LE: lbu       $1, 0($1)               # encoding: [0x00,0x00,0x21,0x90]
  # CHECK-LE: sll       $8, $8, 8               # encoding: [0x00,0x42,0x08,0x00]
  # CHECK-LE: or        $8, $8, $1              # encoding: [0x25,0x40,0x01,0x01]

usw_imm: # CHECK-LABEL: usw_imm:
  usw $8, 0
  # CHECK-BE:   swl     $8, 0($zero)            # encoding: [0xa8,0x08,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($zero)            # encoding: [0xb8,0x08,0x00,0x03]
  # CHECK-LE:   swl     $8, 3($zero)            # encoding: [0x03,0x00,0x08,0xa8]
  # CHECK-LE:   swr     $8, 0($zero)            # encoding: [0x00,0x00,0x08,0xb8]

  usw $8, 2
  # CHECK-BE:   swl     $8, 2($zero)            # encoding: [0xa8,0x08,0x00,0x02]
  # CHECK-BE:   swr     $8, 5($zero)            # encoding: [0xb8,0x08,0x00,0x05]
  # CHECK-LE:   swl     $8, 5($zero)            # encoding: [0x05,0x00,0x08,0xa8]
  # CHECK-LE:   swr     $8, 2($zero)            # encoding: [0x02,0x00,0x08,0xb8]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  usw $8, 0x8000
  # CHECK-BE:   ori     $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
  # CHECK-BE:   move     $1, $1                 # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   ori     $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
  # CHECK-LE:   move     $1, $1                 # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, -0x8000
  # CHECK-BE:   swl     $8, -32768($zero)       # encoding: [0xa8,0x08,0x80,0x00]
  # CHECK-BE:   swr     $8, -32765($zero)       # encoding: [0xb8,0x08,0x80,0x03]
  # CHECK-LE:   swl     $8, -32765($zero)       # encoding: [0x03,0x80,0x08,0xa8]
  # CHECK-LE:   swr     $8, -32768($zero)       # encoding: [0x00,0x80,0x08,0xb8]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  usw $8, 0x10000
  # CHECK-BE:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE:   move    $1, $1                  # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE:   move    $1, $1                  # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  usw $8, 0x18888
  # CHECK-BE:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE:   ori     $1, $1, 34952           # encoding: [0x34,0x21,0x88,0x88]
  # CHECK-BE:   move    $1, $1                  # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE:   ori     $1, $1, 34952           # encoding: [0x88,0x88,0x21,0x34]
  # CHECK-LE:   move    $1, $1                  # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  # FIXME: Remove the identity moves (move $1, $1) coming from loadImmediate
  usw $8, -32769
  # CHECK-BE:   lui     $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
  # CHECK-BE:   ori     $1, $1, 32767           # encoding: [0x34,0x21,0x7f,0xff]
  # CHECK-BE:   move    $1, $1                  # encoding: [0x00,0x20,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
  # CHECK-LE:   ori     $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
  # CHECK-LE:   move    $1, $1                  # encoding: [0x21,0x08,0x20,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, 32767
  # CHECK-BE:   addiu   $1, $zero, 32767        # encoding: [0x24,0x01,0x7f,0xff]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   addiu   $1, $zero, 32767        # encoding: [0xff,0x7f,0x01,0x24]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

usw_reg: # CHECK-LABEL: usw_reg:
  usw $8, 0($9)
  # CHECK-BE:   swl     $8, 0($9)               # encoding: [0xa9,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($9)               # encoding: [0xb9,0x28,0x00,0x03]
  # CHECK-LE:   swl     $8, 3($9)               # encoding: [0x03,0x00,0x28,0xa9]
  # CHECK-LE:   swr     $8, 0($9)               # encoding: [0x00,0x00,0x28,0xb9]

  usw $8, 2($9)
  # CHECK-BE:   swl     $8, 2($9)               # encoding: [0xa9,0x28,0x00,0x02]
  # CHECK-BE:   swr     $8, 5($9)               # encoding: [0xb9,0x28,0x00,0x05]
  # CHECK-LE:   swl     $8, 5($9)               # encoding: [0x05,0x00,0x28,0xa9]
  # CHECK-LE:   swr     $8, 2($9)               # encoding: [0x02,0x00,0x28,0xb9]

  usw $8, 0x8000($9)
  # CHECK-BE:   ori     $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
  # CHECK-BE:   addu    $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   ori     $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
  # CHECK-LE:   addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, -0x8000($9)
  # CHECK-BE:   swl     $8, -32768($9)          # encoding: [0xa9,0x28,0x80,0x00]
  # CHECK-BE:   swr     $8, -32765($9)          # encoding: [0xb9,0x28,0x80,0x03]
  # CHECK-LE:   swl     $8, -32765($9)          # encoding: [0x03,0x80,0x28,0xa9]
  # CHECK-LE:   swr     $8, -32768($9)          # encoding: [0x00,0x80,0x28,0xb9]

  usw $8, 0x10000($9)
  # CHECK-BE:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE:   addu    $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE:   addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, 0x18888($9)
  # CHECK-BE:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE:   ori     $1, $1, 34952           # encoding: [0x34,0x21,0x88,0x88]
  # CHECK-BE:   addu    $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE:   ori     $1, $1, 34952           # encoding: [0x88,0x88,0x21,0x34]
  # CHECK-LE:   addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, -32769($9)
  # CHECK-BE:   lui     $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
  # CHECK-BE:   ori     $1, $1, 32767           # encoding: [0x34,0x21,0x7f,0xff]
  # CHECK-BE:   addu    $1, $1, $9              # encoding: [0x00,0x29,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
  # CHECK-LE:   ori     $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
  # CHECK-LE:   addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, 32767($9)
  # CHECK-BE:   addiu   $1, $9, 32767           # encoding: [0x25,0x21,0x7f,0xff]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   addiu   $1, $9, 32767           # encoding: [0xff,0x7f,0x21,0x25]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, 0($8)
  # CHECK-BE:   swl     $8, 0($8)               # encoding: [0xa9,0x08,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($8)               # encoding: [0xb9,0x08,0x00,0x03]
  # CHECK-LE:   swl     $8, 3($8)               # encoding: [0x03,0x00,0x08,0xa9]
  # CHECK-LE:   swr     $8, 0($8)               # encoding: [0x00,0x00,0x08,0xb9]

  usw $8, 2($8)
  # CHECK-BE:   swl     $8, 2($8)               # encoding: [0xa9,0x08,0x00,0x02]
  # CHECK-BE:   swr     $8, 5($8)               # encoding: [0xb9,0x08,0x00,0x05]
  # CHECK-LE:   swl     $8, 5($8)               # encoding: [0x05,0x00,0x08,0xa9]
  # CHECK-LE:   swr     $8, 2($8)               # encoding: [0x02,0x00,0x08,0xb9]

  usw $8, 0x8000($8)
  # CHECK-BE:   ori     $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
  # CHECK-BE:   addu    $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   ori     $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
  # CHECK-LE:   addu    $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, -0x8000($8)
  # CHECK-BE:   swl     $8, -32768($8)          # encoding: [0xa9,0x08,0x80,0x00]
  # CHECK-BE:   swr     $8, -32765($8)          # encoding: [0xb9,0x08,0x80,0x03]
  # CHECK-LE:   swl     $8, -32765($8)          # encoding: [0x03,0x80,0x08,0xa9]
  # CHECK-LE:   swr     $8, -32768($8)          # encoding: [0x00,0x80,0x08,0xb9]

  usw $8, 0x10000($8)
  # CHECK-BE:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE:   addu    $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE:   addu    $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, 0x18888($8)
  # CHECK-BE:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
  # CHECK-BE:   ori     $1, $1, 34952           # encoding: [0x34,0x21,0x88,0x88]
  # CHECK-BE:   addu    $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
  # CHECK-LE:   ori     $1, $1, 34952           # encoding: [0x88,0x88,0x21,0x34]
  # CHECK-LE:   addu    $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, -32769($8)
  # CHECK-BE:   lui     $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
  # CHECK-BE:   ori     $1, $1, 32767           # encoding: [0x34,0x21,0x7f,0xff]
  # CHECK-BE:   addu    $1, $1, $8              # encoding: [0x00,0x28,0x08,0x21]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   lui     $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
  # CHECK-LE:   ori     $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
  # CHECK-LE:   addu    $1, $1, $8              # encoding: [0x21,0x08,0x28,0x00]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

  usw $8, 32767($8)
  # CHECK-BE:   addiu   $1, $8, 32767           # encoding: [0x25,0x01,0x7f,0xff]
  # CHECK-BE:   swl     $8, 0($1)               # encoding: [0xa8,0x28,0x00,0x00]
  # CHECK-BE:   swr     $8, 3($1)               # encoding: [0xb8,0x28,0x00,0x03]
  # CHECK-LE:   addiu   $1, $8, 32767           # encoding: [0xff,0x7f,0x01,0x25]
  # CHECK-LE:   swl     $8, 3($1)               # encoding: [0x03,0x00,0x28,0xa8]
  # CHECK-LE:   swr     $8, 0($1)               # encoding: [0x00,0x00,0x28,0xb8]

1:
  add $4, $4, $4
