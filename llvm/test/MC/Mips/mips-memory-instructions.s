# RUN: llvm-mc %s -triple=mips64el -mcpu=mips64r2 -target-abi=n32 -show-encoding | \
# RUN:   FileCheck %s
# RUN: llvm-mc %s -triple=mips64el -mcpu=mips64r2 -target-abi=n64 -show-encoding | \
# RUN:   FileCheck %s
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for loads and stores.
#------------------------------------------------------------------------------
# Memory store instructions
#------------------------------------------------------------------------------
# CHECK:  sb      $4, 16($5)      # encoding: [0x10,0x00,0xa4,0xa0]
# CHECK:  sc      $4, 16($5)      # encoding: [0x10,0x00,0xa4,0xe0]
# CHECK:  sh      $4, 16($5)      # encoding: [0x10,0x00,0xa4,0xa4]
# CHECK:  sw      $4, 16($5)      # encoding: [0x10,0x00,0xa4,0xac]
# CHECK:  sw      $7,  0($5)      # encoding: [0x00,0x00,0xa7,0xac]
# CHECK:  swc1    $f2, 16($5)     # encoding: [0x10,0x00,0xa2,0xe4]
# CHECK:  swl     $4, 16($5)      # encoding: [0x10,0x00,0xa4,0xa8]
     sb   $4, 16($5)
     sc   $4, 16($5)
     sh   $4, 16($5)
     sw   $4, 16($5)
     sw   $7,   ($5)
     swc1 $f2, 16($5)
     swl  $4, 16($5)

#------------------------------------------------------------------------------
# Memory load instructions
#------------------------------------------------------------------------------

# CHECK:  lb  $4, 4($5)       # encoding: [0x04,0x00,0xa4,0x80]
# CHECK:  lw  $4, 4($5)       # encoding: [0x04,0x00,0xa4,0x8c]
# CHECK:  lbu $4, 4($5)       # encoding: [0x04,0x00,0xa4,0x90]
# CHECK:  lh  $4, 4($5)       # encoding: [0x04,0x00,0xa4,0x84]
# CHECK:  lhu $4, 4($5)       # encoding: [0x04,0x00,0xa4,0x94]
# CHECK:  ll  $4, 4($5)       # encoding: [0x04,0x00,0xa4,0xc0]
# CHECK:  lw  $4, 4($5)       # encoding: [0x04,0x00,0xa4,0x8c]
# CHECK:  lw  $7, 0($7)       # encoding: [0x00,0x00,0xe7,0x8c]
# CHECK:  lw  $2, 16($sp)     # encoding: [0x10,0x00,0xa2,0x8f]

      lb      $4, 4($5)
      lw      $4, 4($5)
      lbu     $4, 4($5)
      lh      $4, 4($5)
      lhu     $4, 4($5)
      ll      $4, 4($5)
      lw      $4, 4($5)
      lw      $7,    ($7)
      lw      $2, 16($sp)

# Omitted offsets, parenthesized expressions, and implicit zero bases.
# CHECK: lw $2, 0($sp)       # encoding: [0x00,0x00,0xa2,0x8f]
# CHECK: lw $2, 12($3)       # encoding: [0x0c,0x00,0x62,0x8c]
# CHECK: lw $2, 12($zero)    # encoding: [0x0c,0x00,0x02,0x8c]
# CHECK: synci 0($3)         # encoding: [0x00,0x00,0x7f,0x04]
# CHECK: synci 12($3)        # encoding: [0x0c,0x00,0x7f,0x04]
# CHECK: synci 12($zero)     # encoding: [0x0c,0x00,0x1f,0x04]

      lw      $2, ($sp)
      lw      $2, (4 + 8)($3)
      lw      $2, 12
      synci   ($3)
      synci   (4 + 8)($3)
      synci   12
