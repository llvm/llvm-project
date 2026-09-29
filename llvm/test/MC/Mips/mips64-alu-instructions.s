# RUN: llvm-mc %s -triple=mips64el-unknown-linux -show-encoding -mcpu=mips64r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
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
# CHECK:  ori    $4, $5, 17767   # encoding: [0x67,0x45,0xa4,0x34]
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
# CHECK:  xori    $9, $6, 17767  # encoding: [0x67,0x45,0xc9,0x38]
# CHECK:  xori   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x38]
# CHECK:  wsbh   $6, $7          # encoding: [0xa0,0x30,0x07,0x7c]
# CHECK:  not    $7, $8          # encoding: [0x27,0x38,0x00,0x01]
     and    $9,  $6, $7
     and    $9,  $6, 17767
     andi   $9,  $6, 17767
     clo    $6,  $7
     clz    $6,  $7
     ins    $19, $9, 6,7
     nor    $9,  $6, $7
     or     $3,  $3, $5
     or     $4,  $5, 17767
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

# CHECK:  dadd    $9, $6, $7      # encoding: [0x2c,0x48,0xc7,0x00]
# CHECK:  daddi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x60]
# CHECK:  daddiu  $9, $6, -15001  # encoding: [0x67,0xc5,0xc9,0x64]
# CHECK:  daddi   $9, $6, 17767   # encoding: [0x67,0x45,0xc9,0x60]
# CHECK:  daddi   $9, $9, 17767   # encoding: [0x67,0x45,0x29,0x61]
# CHECK:  daddiu  $9, $6, -15001  # encoding: [0x67,0xc5,0xc9,0x64]
# CHECK:  daddiu  $9, $9, -15001  # encoding: [0x67,0xc5,0x29,0x65]
# CHECK:  daddu   $9, $6, $7      # encoding: [0x2d,0x48,0xc7,0x00]
# CHECK:  drotr   $9, $6, 20      # encoding: [0x3a,0x4d,0x26,0x00]
# CHECK:  drotr32 $9, $6, 20      # encoding: [0x3e,0x4d,0x26,0x00]
# CHECK:  madd   $6, $7          # encoding: [0x00,0x00,0xc7,0x70]
# CHECK:  maddu  $6, $7          # encoding: [0x01,0x00,0xc7,0x70]
# CHECK:  msub   $6, $7          # encoding: [0x04,0x00,0xc7,0x70]
# CHECK:  msubu  $6, $7          # encoding: [0x05,0x00,0xc7,0x70]
# CHECK:  mult   $3, $5          # encoding: [0x18,0x00,0x65,0x00]
# CHECK:  multu  $3, $5          # encoding: [0x19,0x00,0x65,0x00]
# CHECK:  dsub    $9, $6, $7     # encoding: [0x2e,0x48,0xc7,0x00]
# CHECK:  dsubu   $4, $3, $5     # encoding: [0x2f,0x20,0x65,0x00]
# CHECK:  daddiu  $9, $6, -17767 # encoding: [0x99,0xba,0xc9,0x64]
# CHECK:  move    $7, $8         # encoding: [0x25,0x38,0x00,0x01]
# CHECK:  .set    push
# CHECK:  .set    mips32r2
# CHECK:  rdhwr   $5, $29
# CHECK:  .set    pop            # encoding: [0x3b,0xe8,0x05,0x7c]

    dadd    $9,$6,$7
    dadd    $9,$6,17767
    daddu   $9,$6,-15001
    daddi   $9,$6,17767
    daddi   $9,17767
    daddiu  $9,$6,-15001
    daddiu  $9,-15001
    daddu   $9,$6,$7
    drotr   $9, $6, 20
    drotr32 $9, $6, 20
    madd   $6,$7
    maddu  $6,$7
    msub   $6,$7
    msubu  $6,$7
    mult   $3,$5
    multu  $3,$5
    dsub    $9,$6,$7
    dsubu   $4,$3,$5
    dsubu   $9,$6,17767
    move   $7,$8
    rdhwr   $5, $29

#------------------------------------------------------------------------------
# Shortcuts for arithmetic instructions
#------------------------------------------------------------------------------

# CHECK:	and	$9, $9, $3	# encoding: [0x24,0x48,0x23,0x01]
# CHECK:	dadd	$9, $9, $3	# encoding: [0x2c,0x48,0x23,0x01]
# CHECK:	daddu	$9, $9, $3	# encoding: [0x2d,0x48,0x23,0x01]
# CHECK:	daddi	$9, $9, 10	# encoding: [0x0a,0x00,0x29,0x61]
# CHECK:	daddiu	$9, $9, 10	# encoding: [0x0a,0x00,0x29,0x65]
# CHECK:	dsub	$9, $9, $3	# encoding: [0x2e,0x48,0x23,0x01]
# CHECK:	dsubu	$9, $9, $3	# encoding: [0x2f,0x48,0x23,0x01]
# CHECK:	daddi	$9, $9, -10	# encoding: [0xf6,0xff,0x29,0x61]
# CHECK:	daddiu	$9, $9, -10	# encoding: [0xf6,0xff,0x29,0x65]
# CHECK:	or	$9, $9, $3	# encoding: [0x25,0x48,0x23,0x01]
# CHECK:	xor	$9, $9, $3	# encoding: [0x26,0x48,0x23,0x01]
	and	$9, $3
	dadd	$9, $3
	daddu	$9, $3
	dadd	$9, 10
	daddu	$9, 10
	dsub	$9, $3
	dsubu	$9, $3
	dsub	$9, 10
	dsubu	$9, 10
	or	$9, $3
	xor	$9, $3

#------------------------------------------------------------------------------
# Did you know that GAS supports complex arithmetic expressions in assembly?
#------------------------------------------------------------------------------
# CHECK:	daddiu	$9, $3, 32	# encoding: [0x20,0x00,0x69,0x64]
# CHECK:	daddiu	$9, $3, 32	# encoding: [0x20,0x00,0x69,0x64]
# CHECK:	daddiu	$9, $3, -32	# encoding: [0xe0,0xff,0x69,0x64]
# CHECK:	daddiu	$9, $3, -32	# encoding: [0xe0,0xff,0x69,0x64] 
	daddiu	$9, $3, 8 * 4
	daddiu	$9, $3, (8 * 4)
	dsubu	$9, $3, 8 * 4
	dsubu	$9, $3, (8 * 4)
