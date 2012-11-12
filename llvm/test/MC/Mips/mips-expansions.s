# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for macro instructions
# CHECK: .section __TEXT,__text,regular,pure_instructions
#------------------------------------------------------------------------------
# Load immediate instructions
#------------------------------------------------------------------------------
# CHECK: ori     $5, $zero, 123      # encoding: [0x7b,0x00,0x05,0x34]
# CHECK: addiu   $6, $zero, -2345    # encoding: [0xd7,0xf6,0x06,0x24]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addiu   $4, $zero, 20       # encoding: [0x14,0x00,0x04,0x24]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addiu   $4, $5, 20          # encoding: [0x14,0x00,0xa4,0x24]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addu    $7, $7, $8          # encoding: [0x21,0x38,0xe8,0x00]

    li $5,123
    li $6,-2345
    li $7,65538

    la $a0, 20
    la $7,65538
    la $a0, 20($a1)
    la $7,65538($8)
