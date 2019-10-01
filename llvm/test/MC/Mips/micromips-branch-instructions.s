# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: | FileCheck %s -check-prefix=CHECK-EL
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: | FileCheck %s -check-prefix=CHECK-EB
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
#------------------------------------------------------------------------------
# Branch Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: b   1332             # encoding: [0x00,0x94,0x9a,0x02]
# CHECK-EL: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: beq $9, $6, 1332     # encoding: [0xc9,0x94,0x9a,0x02]
# CHECK-EL-NEXT: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: bgez $6, 1332        # encoding: [0x46,0x40,0x9a,0x02]
# CHECK-EL-NEXT: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: bgezal $6, 1332      # encoding: [0x66,0x40,0x9a,0x02]
# CHECK-EL: nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: bltzal $6, 1332      # encoding: [0x26,0x40,0x9a,0x02]
# CHECK-EL: nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: bgtz $6, 1332        # encoding: [0xc6,0x40,0x9a,0x02]
# CHECK-EL-NEXT: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: blez $6, 1332        # encoding: [0x86,0x40,0x9a,0x02]
# CHECK-EL-NEXT: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: bne $9, $6, 1332     # encoding: [0xc9,0xb4,0x9a,0x02]
# CHECK-EL-NEXT: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: bal 1332             # encoding: [0x60,0x40,0x9a,0x02]
# CHECK-EL: nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: bltz $6, 1332        # encoding: [0x06,0x40,0x9a,0x02]
# CHECK-EL-NEXT: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: bgezals $6, 1332     # encoding: [0x66,0x42,0x9a,0x02]
# CHECK-EL: nop                  # encoding: [0x00,0x0c]
# CHECK-EL: bltzals $6, 1332     # encoding: [0x26,0x42,0x9a,0x02]
# CHECK-EL: nop                  # encoding: [0x00,0x0c]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: b   1332             # encoding: [0x94,0x00,0x02,0x9a]
# CHECK-EB: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: beq $9, $6, 1332     # encoding: [0x94,0xc9,0x02,0x9a]
# CHECK-EB-NEXT: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: bgez $6, 1332        # encoding: [0x40,0x46,0x02,0x9a]
# CHECK-EB-NEXT: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: bgezal $6, 1332      # encoding: [0x40,0x66,0x02,0x9a]
# CHECK-EB: nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: bltzal $6, 1332      # encoding: [0x40,0x26,0x02,0x9a]
# CHECK-EB: nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: bgtz $6, 1332        # encoding: [0x40,0xc6,0x02,0x9a]
# CHECK-EB-NEXT: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: blez $6, 1332        # encoding: [0x40,0x86,0x02,0x9a]
# CHECK-EB-NEXT: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: bne $9, $6, 1332     # encoding: [0xb4,0xc9,0x02,0x9a]
# CHECK-EB-NEXT: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: bal 1332             # encoding: [0x40,0x60,0x02,0x9a]
# CHECK-EB: nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: bltz $6, 1332        # encoding: [0x40,0x06,0x02,0x9a]
# CHECK-EB-NEXT: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: bgezals $6, 1332     # encoding: [0x42,0x66,0x02,0x9a]
# CHECK-EB: nop                  # encoding: [0x0c,0x00]
# CHECK-EB: bltzals $6, 1332     # encoding: [0x42,0x26,0x02,0x9a]
# CHECK-EB: nop                  # encoding: [0x0c,0x00]

    .text
    .type main, @function
    .set micromips
main:
     b      1332
     beq    $9,$6,1332
     bgez   $6,1332
     bgezal $6,1332
     bltzal $6,1332
     bgtz   $6,1332
     blez   $6,1332
     bne    $9,$6,1332
     bal    1332
     bltz   $6,1332
     bgezals $6,1332
     bltzals $6,1332
