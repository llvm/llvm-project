# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
#------------------------------------------------------------------------------
# Arithmetic and Logical Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: add   $9, $6, $7      # encoding: [0xe6,0x00,0x10,0x49]
# CHECK-EL: addi  $9, $6, 17767   # encoding: [0x26,0x11,0x67,0x45]
# CHECK-EL: addiu $9, $6, -15001  # encoding: [0x26,0x31,0x67,0xc5]
# CHECK-EL: addi  $9, $6, 17767   # encoding: [0x26,0x11,0x67,0x45]
# CHECK-EL: addiu $9, $6, -15001  # encoding: [0x26,0x31,0x67,0xc5]
# CHECK-EL: addu  $9, $6, $7      # encoding: [0xe6,0x00,0x50,0x49]
# CHECK-EL: sub   $9, $6, $7      # encoding: [0xe6,0x00,0x90,0x49]
# CHECK-EL: subu  $4, $3, $5      # encoding: [0xa3,0x00,0xd0,0x21]
# CHECK-EL: neg   $6, $7          # encoding: [0xe0,0x00,0x90,0x31]
# CHECK-EL: negu  $6, $7          # encoding: [0xe0,0x00,0xd0,0x31]
# CHECK-EL: slt    $3, $3, $5     # encoding: [0xa3,0x00,0x50,0x1b]
# CHECK-EL: slti   $3, $3, 103    # encoding: [0x63,0x90,0x67,0x00]
# CHECK-EL: slti   $3, $3, 103    # encoding: [0x63,0x90,0x67,0x00]
# CHECK-EL: sltiu  $3, $3, 103    # encoding: [0x63,0xb0,0x67,0x00]
# CHECK-EL: sltu   $3, $3, $5     # encoding: [0xa3,0x00,0x90,0x1b]
# CHECK-EL: lui    $9, 17767      # encoding: [0xa9,0x41,0x67,0x45]
# CHECK-EL: and    $9, $6, $7     # encoding: [0xe6,0x00,0x50,0x4a]
# CHECK-EL: andi   $9, $6, 17767  # encoding: [0x26,0xd1,0x67,0x45]
# CHECK-EL: andi   $9, $6, 17767  # encoding: [0x26,0xd1,0x67,0x45]
# CHECK-EL: or     $3, $4, $5     # encoding: [0xa4,0x00,0x90,0x1a]
# CHECK-EL: ori    $9, $6, 17767  # encoding: [0x26,0x51,0x67,0x45]
# CHECK-EL: xor    $3, $3, $5     # encoding: [0xa3,0x00,0x10,0x1b]
# CHECK-EL: xori   $9, $6, 17767  # encoding: [0x26,0x71,0x67,0x45]
# CHECK-EL: xori   $9, $6, 17767  # encoding: [0x26,0x71,0x67,0x45]
# CHECK-EL: nor    $9, $6, $7     # encoding: [0xe6,0x00,0xd0,0x4a]
# CHECK-EL: not    $7, $8         # encoding: [0x08,0x00,0xd0,0x3a]
# CHECK-EL: mul    $9, $6, $7     # encoding: [0xe6,0x00,0x10,0x4a]
# CHECK-EL: mult   $9, $7         # encoding: [0xe9,0x00,0x3c,0x8b]
# CHECK-EL: multu  $9, $7         # encoding: [0xe9,0x00,0x3c,0x9b]
# CHECK-EL: div    $zero, $9, $7  # encoding: [0xe9,0x00,0x3c,0xab]
# CHECK-EL: divu   $zero, $9, $7  # encoding: [0xe9,0x00,0x3c,0xbb]
# CHECK-EL: addiupc $2, 20        # encoding: [0x00,0x79,0x05,0x00]
# CHECK-EL: addiupc $7, 16777212  # encoding: [0xbf,0x7b,0xff,0xff]
# CHECK-EL: addiupc $7, -16777216 # encoding: [0xc0,0x7b,0x00,0x00]
# CHECK-EL: andi $3, $3, 5        # encoding: [0x63,0xd0,0x05,0x00]
# CHECK-EL: ori $3, $4, 5         # encoding: [0x64,0x50,0x05,0x00]
# CHECK-EL: ori $3, $3, 5         # encoding: [0x63,0x50,0x05,0x00]
# CHECK-EL: xori $3, $3, 5        # encoding: [0x63,0x70,0x05,0x00]
# CHECK-EL: not $3, $4            # encoding: [0x04,0x00,0xd0,0x1a]

#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: add $9, $6, $7        # encoding: [0x00,0xe6,0x49,0x10]
# CHECK-EB: addi  $9, $6, 17767   # encoding: [0x11,0x26,0x45,0x67]
# CHECK-EB: addiu $9, $6, -15001  # encoding: [0x31,0x26,0xc5,0x67]
# CHECK-EB: addi  $9, $6, 17767   # encoding: [0x11,0x26,0x45,0x67]
# CHECK-EB: addiu $9, $6, -15001  # encoding: [0x31,0x26,0xc5,0x67]
# CHECK-EB: addu  $9, $6, $7      # encoding: [0x00,0xe6,0x49,0x50]
# CHECK-EB: sub $9, $6, $7        # encoding: [0x00,0xe6,0x49,0x90]
# CHECK-EB: subu  $4, $3, $5      # encoding: [0x00,0xa3,0x21,0xd0]
# CHECK-EB: neg $6, $7            # encoding: [0x00,0xe0,0x31,0x90]
# CHECK-EB: negu  $6, $7          # encoding: [0x00,0xe0,0x31,0xd0]
# CHECK-EB: slt $3, $3, $5        # encoding: [0x00,0xa3,0x1b,0x50]
# CHECK-EB: slti  $3, $3, 103     # encoding: [0x90,0x63,0x00,0x67]
# CHECK-EB: slti  $3, $3, 103     # encoding: [0x90,0x63,0x00,0x67]
# CHECK-EB: sltiu $3, $3, 103     # encoding: [0xb0,0x63,0x00,0x67]
# CHECK-EB: sltu  $3, $3, $5      # encoding: [0x00,0xa3,0x1b,0x90]
# CHECK-EB: lui $9, 17767         # encoding: [0x41,0xa9,0x45,0x67]
# CHECK-EB: and $9, $6, $7        # encoding: [0x00,0xe6,0x4a,0x50]
# CHECK-EB:  andi  $9, $6, 17767  # encoding: [0xd1,0x26,0x45,0x67]
# CHECK-EB:  andi  $9, $6, 17767  # encoding: [0xd1,0x26,0x45,0x67]
# CHECK-EB:  or  $3, $4, $5       # encoding: [0x00,0xa4,0x1a,0x90]
# CHECK-EB:  ori $9, $6, 17767    # encoding: [0x51,0x26,0x45,0x67]
# CHECK-EB:  xor $3, $3, $5       # encoding: [0x00,0xa3,0x1b,0x10]
# CHECK-EB:  xori  $9, $6, 17767  # encoding: [0x71,0x26,0x45,0x67]
# CHECK-EB:  xori  $9, $6, 17767  # encoding: [0x71,0x26,0x45,0x67]
# CHECK-EB:  nor $9, $6, $7       # encoding: [0x00,0xe6,0x4a,0xd0]
# CHECK-EB:  not $7, $8           # encoding: [0x00,0x08,0x3a,0xd0]
# CHECK-EB:  mul $9, $6, $7       # encoding: [0x00,0xe6,0x4a,0x10]
# CHECK-EB:  mult  $9, $7         # encoding: [0x00,0xe9,0x8b,0x3c]
# CHECK-EB:  multu $9, $7         # encoding: [0x00,0xe9,0x9b,0x3c]
# CHECK-EB: div  $zero, $9, $7    # encoding: [0x00,0xe9,0xab,0x3c]
# CHECK-EB: divu $zero, $9, $7    # encoding: [0x00,0xe9,0xbb,0x3c]
# CHECK-EB: addiupc $2, 20        # encoding: [0x79,0x00,0x00,0x05]
# CHECK-EB: addiupc $7, 16777212  # encoding: [0x7b,0xbf,0xff,0xff]
# CHECK-EB: addiupc $7, -16777216 # encoding: [0x7b,0xc0,0x00,0x00]
# CHECK-EB: andi $3, $3, 5        # encoding: [0xd0,0x63,0x00,0x05]
# CHECK-EB: ori $3, $4, 5         # encoding: [0x50,0x64,0x00,0x05]
# CHECK-EB: ori $3, $3, 5         # encoding: [0x50,0x63,0x00,0x05]
# CHECK-EB: xori $3, $3, 5        # encoding: [0x70,0x63,0x00,0x05]
# CHECK-EB: not $3, $4            # encoding: [0x00,0x04,0x1a,0xd0]

    add    $9, $6, $7
    add    $9, $6, 17767
    addu   $9, $6, -15001
    addi   $9, $6, 17767
    addiu  $9, $6,-15001
    addu   $9, $6, $7
    sub    $9, $6, $7
    subu   $4, $3, $5
    neg    $6, $7
    negu   $6, $7
    move   $7, $8
    slt    $3, $3, $5
    slt    $3, $3, 103
    slti   $3, $3, 103
    sltiu  $3, $3, 103
    sltu   $3, $3, $5
    lui    $9, 17767
    and    $9, $6, $7
    and    $9, $6, 17767
    andi   $9, $6, 17767
    or     $3, $4, $5
    ori    $9, $6, 17767
    xor    $3, $3, $5
    xor    $9, $6, 17767
    xori   $9, $6, 17767
    nor    $9, $6, $7
    nor    $7, $8, $zero
    mul    $9, $6, $7
    mult   $9, $7
    multu  $9, $7
    div    $0, $9, $7
    divu   $0, $9, $7
    addiupc $2, 20
    addiupc $7, 16777212
    addiupc $7, -16777216
    and    $3, 5
    or     $3, $4, 5
    or     $3, 5
    xor    $3, 5
    not    $3, $4
