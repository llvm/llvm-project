# RUN: llvm-mc -arch=mips -mcpu=mips32r2 -mattr=+mt -show-encoding < %s | FileCheck %s

# The selector value and register values here are marked as reserved in the
# documentation, but GAS accepts them without warning.
  mftr  $31, $31, 1, 1, 0       # CHECK: mftr  $ra, $ra, 1, 1, 0   # encoding: [0x41,0x1f,0xf8,0x21]
  mttr  $31, $31, 1, 1, 0       # CHECK: mttr  $ra, $ra, 1, 1, 0   # encoding: [0x41,0x9f,0xf8,0x21]
  mftr  $31, $13, 1, 6, 0       # CHECK: mftr  $ra, $13, 1, 6, 0   # encoding: [0x41,0x0d,0xf8,0x26]
  mttr  $31, $13, 1, 6, 0       # CHECK: mttr  $ra, $13, 1, 6, 0   # encoding: [0x41,0x9f,0x68,0x26]
