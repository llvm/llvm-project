# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN: -mattr=micromips | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding \
# RUN: -mattr=micromips | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for loads and stores.
#------------------------------------------------------------------------------
# Load and Store unaligned instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: lwl $4, 16($5)   # encoding: [0x85,0x60,0x10,0x00]
# CHECK-EL: lwr $4, 16($5)   # encoding: [0x85,0x60,0x10,0x10]
# CHECK-EL: swl $4, 16($5)   # encoding: [0x85,0x60,0x10,0x80]
# CHECK-EL: swr $4, 16($5)   # encoding: [0x85,0x60,0x10,0x90]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: lwl $4, 16($5)   # encoding: [0x60,0x85,0x00,0x10]
# CHECK-EB: lwr $4, 16($5)   # encoding: [0x60,0x85,0x10,0x10]
# CHECK-EB: swl $4, 16($5)   # encoding: [0x60,0x85,0x80,0x10]
# CHECK-EB: swr $4, 16($5)   # encoding: [0x60,0x85,0x90,0x10]
     lwl  $4, 16($5)
     lwr  $4, 16($5)
     swl  $4, 16($5)
     swr  $4, 16($5)
