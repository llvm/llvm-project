# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for Multiply Add/Sub instructions.
#------------------------------------------------------------------------------
# Multiply Add/Sub Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: madd   $4, $5    # encoding: [0xa4,0x00,0x3c,0xcb]
# CHECK-EL: maddu  $4, $5    # encoding: [0xa4,0x00,0x3c,0xdb]
# CHECK-EL: msub   $4, $5    # encoding: [0xa4,0x00,0x3c,0xeb]
# CHECK-EL: msubu  $4, $5    # encoding: [0xa4,0x00,0x3c,0xfb]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: madd   $4, $5    # encoding: [0x00,0xa4,0xcb,0x3c]
# CHECK-EB: maddu  $4, $5    # encoding: [0x00,0xa4,0xdb,0x3c]
# CHECK-EB: msub   $4, $5    # encoding: [0x00,0xa4,0xeb,0x3c]
# CHECK-EB: msubu  $4, $5    # encoding: [0x00,0xa4,0xfb,0x3c]
    madd     $4, $5
    maddu    $4, $5
    msub     $4, $5
    msubu    $4, $5
