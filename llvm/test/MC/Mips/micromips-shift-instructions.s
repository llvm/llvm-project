# RUN: llvm-mc %s -triple=mipsel -show-encoding -mcpu=mips32r2 -mattr=micromips | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mcpu=mips32r2 -mattr=micromips | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for shift instructions.
#------------------------------------------------------------------------------
# Shift Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: sll    $4, $3, 7      # encoding: [0x83,0x00,0x00,0x38]
# CHECK-EL: sllv   $2, $3, $5     # encoding: [0x65,0x00,0x10,0x10]
# CHECK-EL: sra    $4, $3, 7      # encoding: [0x83,0x00,0x80,0x38]
# CHECK-EL: srav   $2, $3, $5     # encoding: [0x65,0x00,0x90,0x10]
# CHECK-EL: srl    $4, $3, 7      # encoding: [0x83,0x00,0x40,0x38]
# CHECK-EL: srlv   $2, $3, $5     # encoding: [0x65,0x00,0x50,0x10]
# CHECK-EL: rotr   $9, $6, 7      # encoding: [0x26,0x01,0xc0,0x38]
# CHECK-EL: rotrv  $9, $6, $7     # encoding: [0xc7,0x00,0xd0,0x48]
# CHECK-EL: sllv   $2, $3, $5     # encoding: [0x65,0x00,0x10,0x10]
# CHECK-EL: srav   $2, $3, $5     # encoding: [0x65,0x00,0x90,0x10]
# CHECK-EL: srlv   $2, $3, $5     # encoding: [0x65,0x00,0x50,0x10]
# CHECK-EL: sllv   $2, $2, $3     # encoding: [0x43,0x00,0x10,0x10]
# CHECK-EL: srav   $2, $2, $3     # encoding: [0x43,0x00,0x90,0x10]
# CHECK-EL: srlv   $2, $2, $3     # encoding: [0x43,0x00,0x50,0x10]
# CHECK-EL: sll    $3, $3, 7      # encoding: [0x63,0x00,0x00,0x38]
# CHECK-EL: sra    $3, $3, 7      # encoding: [0x63,0x00,0x80,0x38]
# CHECK-EL: srl    $3, $3, 7      # encoding: [0x63,0x00,0x40,0x38]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: sll $4, $3, 7         # encoding: [0x00,0x83,0x38,0x00]
# CHECK-EB: sllv  $2, $3, $5      # encoding: [0x00,0x65,0x10,0x10]
# CHECK-EB: sra $4, $3, 7         # encoding: [0x00,0x83,0x38,0x80]
# CHECK-EB: srav  $2, $3, $5      # encoding: [0x00,0x65,0x10,0x90]
# CHECK-EB: srl $4, $3, 7         # encoding: [0x00,0x83,0x38,0x40]
# CHECK-EB: srlv  $2, $3, $5      # encoding: [0x00,0x65,0x10,0x50]
# CHECK-EB: rotr  $9, $6, 7       # encoding: [0x01,0x26,0x38,0xc0]
# CHECK-EB: rotrv $9, $6, $7      # encoding: [0x00,0xc7,0x48,0xd0]
# CHECK-EB: sllv $2, $3, $5       # encoding: [0x00,0x65,0x10,0x10]
# CHECK-EB: srav $2, $3, $5       # encoding: [0x00,0x65,0x10,0x90]
# CHECK-EB: srlv $2, $3, $5       # encoding: [0x00,0x65,0x10,0x50]
# CHECK-EB: sllv $2, $2, $3       # encoding: [0x00,0x43,0x10,0x10]
# CHECK-EB: srav $2, $2, $3       # encoding: [0x00,0x43,0x10,0x90]
# CHECK-EB: srlv $2, $2, $3       # encoding: [0x00,0x43,0x10,0x50]
# CHECK-EB: sll $3, $3, 7         # encoding: [0x00,0x63,0x38,0x00]
# CHECK-EB: sra $3, $3, 7         # encoding: [0x00,0x63,0x38,0x80]
# CHECK-EB: srl $3, $3, 7         # encoding: [0x00,0x63,0x38,0x40]
     sll    $4, $3, 7
     sllv   $2, $3, $5
     sra    $4, $3, 7
     srav   $2, $3, $5
     srl    $4, $3, 7
     srlv   $2, $3, $5
     rotr   $9, $6, 7
     rotrv  $9, $6, $7
     sll    $2, $3, $5
     sra    $2, $3, $5
     srl    $2, $3, $5
     sll    $2, $3
     sra    $2, $3
     srl    $2, $3
     sll    $3, 7
     sra    $3, 7
     srl    $3, 7
