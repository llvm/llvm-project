# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for move conditional instructions.
#------------------------------------------------------------------------------
# Move Conditional
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: movz    $9, $6, $7        # encoding: [0xe6,0x00,0x58,0x48]
# CHECK-EL: movn    $9, $6, $7        # encoding: [0xe6,0x00,0x18,0x48]
# CHECK-EL: movt    $9, $6, $fcc0     # encoding: [0x26,0x55,0x7b,0x09]
# CHECK-EL: movf    $9, $6, $fcc0     # encoding: [0x26,0x55,0x7b,0x01]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: movz    $9, $6, $7        # encoding: [0x00,0xe6,0x48,0x58]
# CHECK-EB: movn    $9, $6, $7        # encoding: [0x00,0xe6,0x48,0x18]
# CHECK-EB: movt    $9, $6, $fcc0     # encoding: [0x55,0x26,0x09,0x7b]
# CHECK-EB: movf    $9, $6, $fcc0     # encoding: [0x55,0x26,0x01,0x7b]
     movz    $9, $6, $7
     movn    $9, $6, $7
     movt    $9, $6, $fcc0
     movf    $9, $6, $fcc0
