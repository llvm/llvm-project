# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN: -mcpu=mips32r2 -mattr=micromips | FileCheck %s
# Check that the assembler can handle the documented syntax
# for macro instructions
#------------------------------------------------------------------------------
# Load immediate instructions
#------------------------------------------------------------------------------
# CHECK: addiu $5, $zero, 123        # encoding: [0xa0,0x30,0x7b,0x00]
# CHECK: addiu $6, $zero, -2345      # encoding: [0xc0,0x30,0xd7,0xf6]
# CHECK: lui   $7, 1                 # encoding: [0xa7,0x41,0x01,0x00]
# CHECK: ori   $7, $7, 2             # encoding: [0xe7,0x50,0x02,0x00]
# CHECK: addiu $4, $zero, 20         # encoding: [0x80,0x30,0x14,0x00]
# CHECK: lui   $7, 1                 # encoding: [0xa7,0x41,0x01,0x00]
# CHECK: ori   $7, $7, 2             # encoding: [0xe7,0x50,0x02,0x00]
# CHECK: addiu $4, $5, 20            # encoding: [0x85,0x30,0x14,0x00]
# CHECK: lui   $7, 1                 # encoding: [0xa7,0x41,0x01,0x00]
# CHECK: ori   $7, $7, 2             # encoding: [0xe7,0x50,0x02,0x00]
# CHECK: addu  $7, $7, $8            # encoding: [0x07,0x01,0x50,0x39]
# CHECK: lui   $10, %hi(symbol)      # encoding: [0xaa'A',0x41'A',0x00,0x00]
# CHECK:                             # fixup A - offset: 0,
# CHECK:                               value: %hi(symbol),
# CHECK:                               kind: fixup_MICROMIPS_HI16
# CHECK: addu  $10, $10, $4          # encoding: [0x8a,0x00,0x50,0x51]
# CHECK: lw    $10, %lo(symbol)($10) # encoding: [0x4a'A',0xfd'A',0x00,0x00]
# CHECK:                             # fixup A - offset: 0,
# CHECK:                               value: %lo(symbol),
# CHECK:                               kind: fixup_MICROMIPS_LO16
# CHECK: lui   $1, %hi(symbol)       # encoding: [0xa1'A',0x41'A',0x00,0x00]
# CHECK:                             # fixup A - offset: 0,
# CHECK:                               value: %hi(symbol),
# CHECK:                               kind: fixup_MICROMIPS_HI16
# CHECK: addu  $1, $1, $9            # encoding: [0x21,0x01,0x50,0x09]
# CHECK: sw    $10, %lo(symbol)($1)  # encoding: [0x41'A',0xf9'A',0x00,0x00]
# CHECK:                             # fixup A - offset: 0,
# CHECK:                               value: %lo(symbol),
# CHECK:                               kind: fixup_MICROMIPS_LO16
# CHECK: lui   $10, 10               # encoding: [0xaa,0x41,0x0a,0x00]
# CHECK: addu  $10, $10, $4          # encoding: [0x8a,0x00,0x50,0x51]
# CHECK: lw    $10, 123($10)         # encoding: [0x4a,0xfd,0x7b,0x00]
# CHECK: lui   $1, 2                 # encoding: [0xa1,0x41,0x02,0x00]
# CHECK: addu  $1, $1, $9            # encoding: [0x21,0x01,0x50,0x09]
# CHECK: sw    $10, -7616($1)        # encoding: [0x41,0xf9,0x40,0xe2]

    li $5,123
    li $6,-2345
    li $7,65538

    la $a0, 20
    la $7,65538
    la $a0, 20($a1)
    la $7,65538($8)

    lw  $t2, symbol($a0)
    sw  $t2, symbol($t1)

    lw  $t2, 655483($a0)
    sw  $t2, 123456($t1)

    lh  $4, 0x8000
# CHECK-LE: lui     $4, 1
# CHECK-LE: lh      $4, -32768($4)

    lh  $4, 0x20004($3)
# CHECK-LE: lui     $4, 2
# CHECK-LE: addu    $4, $4, $3
# CHECK-LE: lh      $4, 4($4)

    lhu  $4, 0x8000
# CHECK-LE: lui     $4, 1
# CHECK-LE: lhu     $4, -32768($4)

    lhu  $4, 0x20004($3)
# CHECK-LE: lui     $4, 2
# CHECK-LE: addu    $4, $4, $3
# CHECK-LE: lhu     $4, 4($4)
