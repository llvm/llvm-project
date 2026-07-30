# RUN: llvm-mc -triple=mips -show-encoding -mcpu=mips1 < %s \
# RUN:   | FileCheck --check-prefixes=ALL,MIPS32 %s
# RUN: llvm-mc -triple=mips -show-encoding -mcpu=mips64 < %s \
# RUN:   | FileCheck --check-prefixes=ALL,MIPS64 %s

sne $4, $5, $6
# ALL:    xor    $4, $5, $6        # encoding: [0x00,0xa6,0x20,0x26]
# ALL:    sltu   $4, $zero, $4     # encoding: [0x00,0x04,0x20,0x2b]
sne $4, $zero, $6
# ALL:    sltu   $4, $zero, $6     # encoding: [0x00,0x06,0x20,0x2b]
sne $4, $5, $zero
# ALL:    sltu   $4, $zero, $5     # encoding: [0x00,0x05,0x20,0x2b]
sne $4, $5, 0
# ALL:    sltu   $4, $zero, $5     # encoding: [0x00,0x05,0x20,0x2b]
sne $4, $zero, 1
# ALL:    addiu  $4, $zero, 1      # encoding: [0x24,0x04,0x00,0x01]
sne $4, $5, -1
# MIPS32: addiu  $4, $5, 1         # encoding: [0x24,0xa4,0x00,0x01]
# MIPS64: daddiu $4, $5, 1         # encoding: [0x64,0xa4,0x00,0x01]
# ALL:    sltu   $4, $zero, $4     # encoding: [0x00,0x04,0x20,0x2b]
sne $4, $5, 1
# ALL:    xori   $4, $5, 1         # encoding: [0x38,0xa4,0x00,0x01]
# ALL:    sltu   $4, $zero, $4     # encoding: [0x00,0x04,0x20,0x2b]
sne $4, $5, 0x10000
# ALL:    lui    $1, 1             # encoding: [0x3c,0x01,0x00,0x01]
# ALL:    xor    $4, $5, $1        # encoding: [0x00,0xa1,0x20,0x26]
# ALL:    sltu   $4, $zero, $4     # encoding: [0x00,0x04,0x20,0x2b]
