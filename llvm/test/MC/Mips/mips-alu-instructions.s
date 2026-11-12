# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
#------------------------------------------------------------------------------
# Logical instructions
#------------------------------------------------------------------------------
# CHECK:  and    $9, $6, $7      # encoding: [0x24,0x48,0xc7,0x00]
# CHECK:  andi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x30]
# CHECK:  andi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x30]
# CHECK:  andi   $9, $9, 17767   # encoding: [0x67,0x45,0x29,0x31]
# CHECK:  clo    $6, $7          # encoding: [0x21,0x30,0xe6,0x70]
# CHECK:  clz    $6, $7          # encoding: [0x20,0x30,0xe6,0x70]
# CHECK:  ins    $19, $9, 6, 7   # encoding: [0x84,0x61,0x33,0x7d]
# CHECK:  nor    $9, $6, $7      # encoding: [0x27,0x48,0xc7,0x00]
# CHECK:  or     $3, $3, $5      # encoding: [0x25,0x18,0x65,0x00]
# CHECK:  ori    $4, $5, 17767   # encoding: [0x67,0x45,0xa4,0x34]
# CHECK:  ori    $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x34]
# CHECK:  ori    $11, $11, 128   # encoding: [0x80,0x00,0x6b,0x35]
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
# CHECK:  xori   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x38]
# CHECK:  xori   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x38]
# CHECK:  xori   $11, $11, 12    # encoding: [0x0c,0x00,0x6b,0x39]
# CHECK:  wsbh   $6, $7          # encoding: [0xa0,0x30,0x07,0x7c]
# CHECK:  not    $7, $8          # encoding: [0x27,0x38,0x00,0x01]
     and    $9,  $6, $7
     and    $9,  $6, 17767
     andi   $9,  $6, 17767
     andi   $9,  17767
     clo    $6,  $7
     clz    $6,  $7
     ins    $19, $9, 6,7
     nor    $9,  $6, $7
     or     $3,  $3, $5
     or     $4,  $5, 17767
     ori    $9,  $6, 17767
     ori    $11, 128
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
     xori   $11, 12
     wsbh   $6,  $7
     not    $7  ,$8

#------------------------------------------------------------------------------
# Arithmetic instructions
#------------------------------------------------------------------------------

# CHECK:  add    $9, $6, $7      # encoding: [0x20,0x48,0xc7,0x00]
# CHECK:  addi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x20]
# CHECK:  addiu  $9, $6, -15001  # encoding: [0x67,0xc5,0xc9,0x24]
# CHECK:  addi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x20]
# CHECK:  addi   $9, $9, 17767   # encoding: [0x67,0x45,0x29,0x21]
# CHECK:  addiu  $9, $6, -15001  # encoding: [0x67,0xc5,0xc9,0x24]
# CHECK:  addiu  $11, $11, 40    # encoding: [0x28,0x00,0x6b,0x25]
# CHECK:  addu   $9, $6, $7      # encoding: [0x21,0x48,0xc7,0x00]
# CHECK:  madd   $6, $7          # encoding: [0x00,0x00,0xc7,0x70]
# CHECK:  maddu  $6, $7          # encoding: [0x01,0x00,0xc7,0x70]
# CHECK:  msub   $6, $7          # encoding: [0x04,0x00,0xc7,0x70]
# CHECK:  msubu  $6, $7          # encoding: [0x05,0x00,0xc7,0x70]
# CHECK:  mult   $3, $5          # encoding: [0x18,0x00,0x65,0x00]
# CHECK:  multu  $3, $5          # encoding: [0x19,0x00,0x65,0x00]
# CHECK:  sub    $9, $6, $7      # encoding: [0x22,0x48,0xc7,0x00]
# CHECK:  addi   $sp, $sp, -56   # encoding: [0xc8,0xff,0xbd,0x23]
# CHECK:  subu   $4, $3, $5      # encoding: [0x23,0x20,0x65,0x00]
# CHECK:  addiu   $sp, $sp, -40  # encoding: [0xd8,0xff,0xbd,0x27]
# CHECK:  neg     $6, $7         # encoding: [0x22,0x30,0x07,0x00]
# CHECK:  negu    $6, $7         # encoding: [0x23,0x30,0x07,0x00]
# CHECK:  move    $7, $8         # encoding: [0x25,0x38,0x00,0x01]
# CHECK:  .set    push
# CHECK:  .set    mips32r2
# CHECK:  rdhwr   $5, $29
# CHECK:  .set    pop            # encoding: [0x3b,0xe8,0x05,0x7c]
    add    $9,$6,$7
    add    $9,$6,17767
    addu   $9,$6,-15001
    addi   $9,$6,17767
    addi   $9,17767
    addiu  $9,$6,-15001
    addiu  $11,40
    addu   $9,$6,$7
    madd   $6,$7
    maddu  $6,$7
    msub   $6,$7
    msubu  $6,$7
    mult   $3,$5
    multu  $3,$5
    sub    $9,$6,$7
    sub    $sp,$sp,56
    subu   $4,$3,$5
    subu    $sp,$sp,40
    neg    $6,$7
    negu   $6,$7
    move   $7,$8
    rdhwr   $5, $29

#------------------------------------------------------------------------------
# Shortcuts for arithmetic instructions
#------------------------------------------------------------------------------

# CHECK:	add	$9, $9, $3	# encoding: [0x20,0x48,0x23,0x01]
# CHECK:	addu	$9, $9, $3	# encoding: [0x21,0x48,0x23,0x01]
# CHECK:	addi	$9, $9, 10	# encoding: [0x0a,0x00,0x29,0x21]
# CHECK:	addiu	$9, $9, 10	# encoding: [0x0a,0x00,0x29,0x25]
# CHECK:	and	$5, $5, $6	# encoding: [0x24,0x28,0xa6,0x00]
# CHECK:	mul	$9, $9, $3	# encoding: [0x02,0x48,0x23,0x71]
# CHECK:	or	$2, $2, $4	# encoding: [0x25,0x10,0x44,0x00]
# CHECK:	sub	$9, $9, $3	# encoding: [0x22,0x48,0x23,0x01]
# CHECK:	subu	$9, $9, $3	# encoding: [0x23,0x48,0x23,0x01]
# CHECK:	addi	$9, $9, -10	# encoding: [0xf6,0xff,0x29,0x21]
# CHECK:	addiu	$9, $9, -10	# encoding: [0xf6,0xff,0x29,0x25]
# CHECK:	xor	$9, $9, $10	# encoding: [0x26,0x48,0x2a,0x01]
	add	$9, $3
	addu	$9, $3
	add	$9, 10
	addu	$9, 10
	and	$5, $6
	mul	$9, $3
	or	$2, $4
	sub	$9, $3
	subu	$9, $3
	sub	$9, 10
	subu	$9, 10
	xor	$9, $10
