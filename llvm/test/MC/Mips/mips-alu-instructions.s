# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
# CHECK: .section __TEXT,__text,regular,pure_instructions
#------------------------------------------------------------------------------
# Logical instructions
#------------------------------------------------------------------------------
# CHECK:  and    $9, $6, $7      # encoding: [0x24,0x48,0xc7,0x00]
# CHECK:  andi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x30]
# CHECK:  andi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x30]
# CHECK:  clo    $6, $7          # encoding: [0x21,0x30,0xe6,0x70]
# CHECK:  clz    $6, $7          # encoding: [0x20,0x30,0xe6,0x70]
# CHECK:  ins    $19, $9, 6, 7   # encoding: [0x84,0x61,0x33,0x7d]
# CHECK:  nor    $9, $6, $7      # encoding: [0x27,0x48,0xc7,0x00]
# CHECK:  or     $3, $3, $5      # encoding: [0x25,0x18,0x65,0x00]
# CHECK:  ori    $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x34]
# CHECK:  rotr   $9, $6, 7       # encoding: [0xc2,0x49,0x26,0x00]
# CHECK:  rotrv  $9, $6, $7      # encoding: [0x46,0x48,0xe6,0x00]
# CHECK:  sll    $4, $3, 7       # encoding: [0xc0,0x21,0x03,0x00]
# CHECK:  sllv   $2, $3, $5      # encoding: [0x04,0x10,0xa3,0x00]
# CHECK:  slt    $3, $3, $5      # encoding: [0x2a,0x18,0x65,0x00]
# CHECK:  slti   $3, $3, 103     # encoding: [0x67,0x00,0x63,0x28]
# CHECK:  slti   $3, $3, 103     # encoding: [0x67,0x00,0x63,0x28]
# CHECK:  sltiu  $3, $3, 103     # encoding: [0x67,0x00,0x63,0x2c]
# CHECK:  sltu   $3, $3, $5      # encoding: [0x2b,0x18,0x65,0x00]
# CHECK:  sra    $4, $3, 7       # encoding: [0xc3,0x21,0x03,0x00]
# CHECK:  srav   $2, $3, $5      # encoding: [0x07,0x10,0xa3,0x00]
# CHECK:  srl    $4, $3, 7       # encoding: [0xc2,0x21,0x03,0x00]
# CHECK:  srlv   $2, $3, $5      # encoding: [0x06,0x10,0xa3,0x00]
# CHECK:  xor    $3, $3, $5      # encoding: [0x26,0x18,0x65,0x00]
# CHECK:  xori    $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x38]
# CHECK:  xori   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x38]
# CHECK:  wsbh   $6, $7          # encoding: [0xa0,0x30,0x07,0x7c]
# CHECK:  nor    $7, $8, $zero   # encoding: [0x27,0x38,0x00,0x01]
     and    $9,  $6, $7
     and    $9,  $6, 17767
     andi   $9,  $6, 17767
     clo    $6,  $7
     clz    $6,  $7
     ins    $19, $9, 6,7
     nor    $9,  $6, $7
     or     $3,  $3, $5
     ori    $9,  $6, 17767
     rotr   $9,  $6, 7
     rotrv  $9,  $6, $7
     sll    $4,  $3, 7
     sllv   $2,  $3, $5
     slt    $3,  $3, $5
     slt    $3,  $3, 103
     slti   $3,  $3, 103
     sltiu  $3,  $3, 103
     sltu   $3,  $3, $5
     sra    $4,  $3, 7
     srav   $2,  $3, $5
     srl    $4,  $3, 7
     srlv   $2,  $3, $5
     xor    $3,  $3, $5
     xor    $9,  $6, 17767
     xori   $9,  $6, 17767
     wsbh   $6,  $7
     not    $7  ,$8

#------------------------------------------------------------------------------
# Arithmetic instructions
#------------------------------------------------------------------------------

# CHECK:  add    $9, $6, $7      # encoding: [0x20,0x48,0xc7,0x00]
# CHECK:  addi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x20]
# CHECK:  addiu  $9, $6, -15001  # encoding: [0x67,0xc5,0xc9,0x24]
# CHECK:  addi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x20]
# CHECK:  addiu  $9, $6, -15001  # encoding: [0x67,0xc5,0xc9,0x24]
# CHECK:  addu   $9, $6, $7      # encoding: [0x21,0x48,0xc7,0x00]
# CHECK:  madd   $6, $7          # encoding: [0x00,0x00,0xc7,0x70]
# CHECK:  maddu  $6, $7          # encoding: [0x01,0x00,0xc7,0x70]
# CHECK:  msub   $6, $7          # encoding: [0x04,0x00,0xc7,0x70]
# CHECK:  msubu  $6, $7          # encoding: [0x05,0x00,0xc7,0x70]
# CHECK:  mult   $3, $5          # encoding: [0x18,0x00,0x65,0x00]
# CHECK:  multu  $3, $5          # encoding: [0x19,0x00,0x65,0x00]
# CHECK:  sub    $9, $6, $7      # encoding: [0x22,0x48,0xc7,0x00]
# CHECK:  subu   $4, $3, $5      # encoding: [0x23,0x20,0x65,0x00]
# CHECK:  sub     $6, $zero, $7  # encoding: [0x22,0x30,0x07,0x00]
# CHECK:  subu    $6, $zero, $7  # encoding: [0x23,0x30,0x07,0x00]
# CHECK:  add     $7, $8, $zero  # encoding: [0x20,0x38,0x00,0x01]
    add    $9,$6,$7
    add    $9,$6,17767
    addu   $9,$6,-15001
    addi   $9,$6,17767
    addiu  $9,$6,-15001
    addu   $9,$6,$7
    madd   $6,$7
    maddu  $6,$7
    msub   $6,$7
    msubu  $6,$7
    mult   $3,$5
    multu  $3,$5
    sub    $9,$6,$7
    subu   $4,$3,$5
    neg    $6,$7
    negu   $6,$7
    move   $7,$8
