# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: | FileCheck %s -check-prefix=CHECK-EL
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: | FileCheck %s -check-prefix=CHECK-EB
# Check that the assembler can handle the documented syntax
# for jump and branch instructions.
#------------------------------------------------------------------------------
# Jump instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: j 1328      # encoding: [0x00,0xd4,0x98,0x02]
# CHECK-EL: nop         # encoding: [0x00,0x0c]
# CHECK-EL: jal 1328    # encoding: [0x00,0xf4,0x98,0x02]
# CHECK-EL: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: jalr $ra, $6 # encoding: [0xe6,0x03,0x3c,0x0f]
# CHECK-EL: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: jr $7       # encoding: [0x07,0x00,0x3c,0x0f]
# CHECK-EL: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: jr $7       # encoding: [0x07,0x00,0x3c,0x0f]
# CHECK-EL: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: jalx 1328   # encoding: [0x00,0xf0,0x4c,0x01]
# CHECK-EL: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: jals 1328         # encoding: [0x00,0x74,0x98,0x02]
# CHECK-EL: nop               # encoding: [0x00,0x0c]
# CHECK-EL: jalrs $ra, $6     # encoding: [0xe6,0x03,0x3c,0x4f]
# CHECK-EL: nop               # encoding: [0x00,0x0c]
# CHECK-EL: jalr $25          # encoding: [0xd9,0x45]
# CHECK-EL: nop               # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: jalr $4, $25      # encoding: [0x99,0x00,0x3c,0x0f]
# CHECK-EL: nop               # encoding: [0x00,0x00,0x00,0x00]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: j 1328      # encoding: [0xd4,0x00,0x02,0x98]
# CHECK-EB: nop         # encoding: [0x0c,0x00]
# CHECK-EB: jal 1328    # encoding: [0xf4,0x00,0x02,0x98]
# CHECK-EB: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: jalr $ra, $6 # encoding: [0x03,0xe6,0x0f,0x3c]
# CHECK-EB: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: jr $7       # encoding: [0x00,0x07,0x0f,0x3c]
# CHECK-EB: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: jr $7       # encoding: [0x00,0x07,0x0f,0x3c]
# CHECK-EB: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: jalx 1328   # encoding: [0xf0,0x00,0x01,0x4c]
# CHECK-EB: nop         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: jals 1328         # encoding: [0x74,0x00,0x02,0x98]
# CHECK-EB: nop               # encoding: [0x0c,0x00]
# CHECK-EB: jalrs $ra, $6     # encoding: [0x03,0xe6,0x4f,0x3c]
# CHECK-EB: nop               # encoding: [0x0c,0x00]
# CHECK-EB: jalr $25          # encoding: [0x45,0xd9]
# CHECK-EB: nop               # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: jalr $4, $25      # encoding: [0x00,0x99,0x0f,0x3c]
# CHECK-EB: nop               # encoding: [0x00,0x00,0x00,0x00]

     j 1328
     jal 1328
     jalr $ra, $6
     jr $7
     j $7
     jalx 1328
     jals 1328
     jalrs $ra, $6
     jal $25
     jal $4, $25
